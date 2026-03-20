# python-implementation of Base + \alpha (RL - Base)
from typing import Optional, Dict, Any, List
import torch
from transformers import (
    Qwen2VLForConditionalGeneration, 
    GenerationMixin
)
import torch.nn.functional as F
from transformers.generation.utils import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList, 
    NoBadWordsLogitsProcessor, 
    SuppressTokensAtBeginLogitsProcessor,
    RepetitionPenaltyLogitsProcessor
)
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from qwen_omni_utils import process_mm_info


USE_AUDIO_IN_VIDEO = False


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)
    

def load_images(messsages):
    images = []
    for message in messsages:
        for item in message:
            if item['type'] == 'image':
                if type(item['image']) == str:
                    image_path = item['image']
                    image = Image.open(image_path)
                    images.append(image)
                else:
                    images.append(item['image'])
    return images


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def extract_text_only_messages(messages):
    """Extract text-only content from multimodal messages."""
    text_only_messages = []
    for message in messages:
        text_only_message = []
        for item in message:
            new_content = ''
            content = item.get('content', [])
            for content_item in content:
                if content_item['type'] == 'text':
                    new_content += content_item['text']
            text_only_message.append({
                'role': item['role'],
                'content': new_content
            })
        text_only_messages.append(text_only_message)
    
    return text_only_messages


class ProxyThinkerWrapper(GenerationMixin):
    def __init__(
        self, 
        base_model: Qwen2VLForConditionalGeneration,
        preprocessor,
        positive_model: Optional[Any] = None,
        negative_model: Optional[Qwen2VLForConditionalGeneration] = None,
        positive_tokenizer: Optional[Any] = None,
        do_torch_compile: Optional[bool] = False,
        input_device_dict: Optional[Dict[str, torch.device]] = None,
        logits_device: Optional[torch.device] = None,
    ):
        self.base_model = base_model
        self.positive_model = positive_model
        self.negative_model = negative_model
        self.preprocessor = preprocessor
        self.positive_tokenizer = positive_tokenizer if positive_tokenizer else preprocessor.tokenizer
        
        self.base_model.eval()
        if self.positive_model is not None:
            self.positive_model.eval()
        
        if self.negative_model is not None:
            self.negative_model.eval()
        
        if do_torch_compile:
            self.base_model = torch.compile(self.base_model)
            if self.positive_model is not None:
                self.positive_model = torch.compile(self.positive_model)
            if self.negative_model is not None:
                self.negative_model = torch.compile(self.negative_model)
        
        if self.positive_model is not None and positive_tokenizer is not None:
            self.tokenizer = positive_tokenizer
        else:
            self.tokenizer = preprocessor.tokenizer
        
        self.input_device_dict = input_device_dict
        if self.input_device_dict is None:
            self.input_device_dict = {
                'base': self.base_model.model.get_input_embeddings().weight.device,
                'positive': self.positive_model.model.get_input_embeddings().weight.device if self.positive_model is not None else None,
                'negative': self.negative_model.model.get_input_embeddings().weight.device if self.negative_model is not None else None
            }
        
        self.logits_device = logits_device
        if self.logits_device is None:
            self.logits_device = self.base_model.lm_head.weight.device
    
    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        analysis_data['tokens'].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data['token_ids'].append(next_tokens)

        for model in next_token_logits_dict.keys():
            analysis_data[f'logits_{model}'].append(next_token_logits_dict[model].unsqueeze(dim=1))

        return analysis_data
        
    def forward(self, base_inputs, positive_inputs, negative_inputs, return_dict=False):
        base_outputs = self.base_model(**base_inputs, return_dict=return_dict)
        positive_outputs, negative_outputs = None, None
        if self.positive_model is not None:
            positive_outputs = self.positive_model(**positive_inputs, return_dict=return_dict)
        
        if self.negative_model is not None:
            negative_outputs = self.negative_model(**negative_inputs, return_dict=return_dict)
        
        return base_outputs, positive_outputs, negative_outputs

    
    def forward_base(self, base_inputs, return_dict=False):
        base_outputs = self.base_model(**base_inputs, return_dict=return_dict)
        return base_outputs
    
    def forward_text_only(self,positive_inputs, negative_inputs, return_dict=False):
        positive_outputs = self.positive_model(**positive_inputs, return_dict=return_dict)
        negative_outputs = self.negative_model(**negative_inputs, return_dict=return_dict)
        return positive_outputs, negative_outputs
    

    def js_divergence(self,p, q):

        p_prob = F.softmax(p, dim=-1)
        q_prob = F.softmax(q, dim=-1)

        m = 0.5 * (p_prob + q_prob)
        
        kl_pm = F.kl_div(torch.log(p_prob + 1e-6), m, reduction='none').sum(dim=-1)
        kl_qm = F.kl_div(torch.log(q_prob + 1e-6), m, reduction='none').sum(dim=-1)
        
        return 0.5 * (kl_pm + kl_qm)
    
    def generate(
        self, 
        inputs,  # input_ids, attention_mask, pixel_values, image_grid_thw
        positive_inputs=None, 
        negative_inputs=None, 
        max_new_tokens: Optional[int] = 1024,
        do_sample: bool = False,
        top_p: float = 1.0,
        top_k: int = 0,
        temperature: float = 0.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        **kwargs
    ):
        batch_size = inputs.input_ids.shape[0]

        base_kwargs = kwargs.copy()
        positive_kwargs = kwargs.copy()
        negative_kwargs = kwargs.copy()

        if self.positive_model is not None and self.negative_model is not None:
            only_pos = torch.tensor(0, device=self.logits_device).unsqueeze(0).repeat(batch_size, 1)
        
        inputs = inputs.to(self.input_device_dict['base'])
        if positive_inputs is None and self.positive_model is not None:
            positive_inputs = deepcopy(inputs).to(self.input_device_dict['positive'])
        
        if negative_inputs is None and self.negative_model is not None:
            negative_inputs = deepcopy(inputs).to(self.input_device_dict['negative'])
            
        unfinished_sequences = torch.ones(inputs.input_ids.shape[0], dtype=torch.long, device=inputs.input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(inputs.input_ids.device)
        
        if return_logits_for_analysis:
            analysis_data = defaultdict(list)
            
        base_kwargs = self._get_initial_cache_position(inputs.input_ids.shape[-1], self.input_device_dict['base'], base_kwargs)
        if positive_inputs is not None:
            positive_kwargs = self._get_initial_cache_position(positive_inputs.input_ids.shape[-1], self.input_device_dict['positive'], positive_kwargs)
        if negative_inputs is not None:
            negative_kwargs = self._get_initial_cache_position(negative_inputs.input_ids.shape[-1], self.input_device_dict['negative'], negative_kwargs)
            
        if "attention_mask" in inputs:
            base_kwargs["attention_mask"] = inputs.pop("attention_mask")
        
        if positive_inputs is not None and "attention_mask" in positive_inputs:
            positive_kwargs["attention_mask"] = positive_inputs.pop("attention_mask")
            
        if negative_inputs is not None and "attention_mask" in negative_inputs:
            negative_kwargs["attention_mask"] = negative_inputs.pop("attention_mask")
        
            
        for step in range(max_new_tokens):
            positive_outputs, negative_outputs = None, None
            base_model_inputs = self.base_model.prepare_inputs_for_generation(**inputs, **base_kwargs)
            positive_model_inputs, negative_model_inputs = None, None
            
            if self.positive_model is not None:
                positive_model_inputs = self.positive_model.prepare_inputs_for_generation(**positive_inputs, **positive_kwargs)
            
            if self.negative_model is not None:
                negative_model_inputs = self.negative_model.prepare_inputs_for_generation(**negative_inputs, **negative_kwargs)
                
            base_outputs, positive_outputs, negative_outputs = self.forward(
                base_model_inputs, positive_model_inputs, negative_model_inputs, return_dict=True
            )
            base_next_token_logits = base_outputs.logits[:, -1, :]


            if positive_outputs is not None:
                positive_next_token_logits = positive_outputs.logits[:, -1, :]
            
            if negative_outputs is not None:
                negative_next_token_logits = negative_outputs.logits[:, -1, :]

            do_trucate_small = False
            larger_vocab_size = base_next_token_logits.shape[-1]
            if positive_outputs is not None and negative_outputs is not None:
                min_vocab_size = min(
                    base_next_token_logits.shape[-1], positive_next_token_logits.shape[-1], negative_next_token_logits.shape[-1]
                )
                base_next_token_logits = base_next_token_logits[:, :min_vocab_size]
                positive_next_token_logits = positive_next_token_logits[:, :min_vocab_size]
                negative_next_token_logits = negative_next_token_logits[:, :min_vocab_size]

                base_next_token_logits = base_next_token_logits.to(self.logits_device)
                positive_next_token_logits = positive_next_token_logits.to(self.logits_device)
                negative_next_token_logits = negative_next_token_logits.to(self.logits_device)

                dis_fun = self.js_divergence
                js_a_c = dis_fun(
                    base_next_token_logits, negative_next_token_logits
                ).reshape(-1, 1)  # [batch_size, 1]
                js_b_c = dis_fun(
                    positive_next_token_logits, negative_next_token_logits
                ).reshape(-1, 1)
                running_alpha = torch.where(
                    js_b_c > js_a_c, 
                    torch.clamp(js_b_c - js_a_c, max=1.0),
                    torch.zeros_like(js_b_c, device=self.logits_device)
                )
                if step <= 5:
                    running_alpha = torch.clamp(running_alpha, max=0.1 * step)

                next_token_logits = (
                    (2.0 - running_alpha) * base_next_token_logits
                    + running_alpha * positive_next_token_logits 
                    - negative_next_token_logits
                )
                if only_pos.max() == 1:
                    next_token_logits = only_pos * positive_next_token_logits + (1 - only_pos) * next_token_logits

                if min_vocab_size < larger_vocab_size:
                    next_token_logits = torch.nn.functional.pad(
                        next_token_logits, 
                        (0, larger_vocab_size - min_vocab_size), 
                        "constant", 
                        float("-inf")
                    )

            else:
                next_token_logits = base_next_token_logits
                
            next_token_logits = next_token_logits.to(inputs['input_ids'].device)
            
            if logits_processor is not None:
                next_token_logits = logits_processor(inputs['input_ids'], next_token_logits)
                
            if temperature != 0.0:
                next_token_logits = next_token_logits / temperature
                
            if top_p < 1.0 or top_k > 0:
                next_token_logits = top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p
                )
                
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                if not do_trucate_small:
                    if positive_outputs is not None and negative_outputs is not None:
                        probs = torch.cat([
                            probs, 
                            torch.zeros(
                                (probs.shape[0], larger_vocab_size - probs.shape[1]), 
                                device=probs.device
                            )
                        ], dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = (
                next_tokens * unfinished_sequences + 
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )
            
            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'proxy': next_token_logits,
                    'base': base_next_token_logits,
                    'positive': positive_next_token_logits,
                    'negative': negative_next_token_logits
                }
                analysis_data = self.update_analysis_data(
                    analysis_data, next_tokens, next_token_logits_dict
                )

            inputs['input_ids'] = torch.cat([inputs['input_ids'], 
                                            next_tokens.unsqueeze(1).to(self.input_device_dict['base'])], 
                                            dim=-1)
            
            if self.positive_model is not None and self.negative_model is not None:
                only_pos = torch.where(
                    next_tokens.unsqueeze(1) == 151668, 
                    torch.tensor(1, device=self.logits_device).unsqueeze(0).repeat(batch_size, 1), 
                    only_pos
                )

            if positive_inputs is not None:
                positive_inputs['input_ids'] = torch.cat([positive_inputs['input_ids'], 
                                                          next_tokens.unsqueeze(1).to(self.input_device_dict['positive'])], dim=-1)
            
            if negative_inputs is not None:
                negative_inputs['input_ids'] = torch.cat([negative_inputs['input_ids'], 
                                                          next_tokens.unsqueeze(1).to(self.input_device_dict['negative'])], dim=-1)
            
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            if positive_outputs is not None:
                positive_kwargs = self._update_model_kwargs_for_generation(positive_outputs, positive_kwargs)
            
            if negative_outputs is not None:
                negative_kwargs = self._update_model_kwargs_for_generation(negative_outputs, negative_kwargs)
            
            if stopping_criteria is not None and torch.all(
                stopping_criteria(inputs['input_ids'], None
            )):
                break
            
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            
            if unfinished_sequences.max() == 0:
                break
            
        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return inputs['input_ids'], analysis_data
        
        return inputs['input_ids']
    
    
@torch.inference_mode()
def generate_completions(
    model_wrapper, 
    processor,
    messages,
    postive_processor=None,
    positive_messages: Optional[List[Dict]] = None,
    negative_messages: Optional[List[Dict]] = None,
    batch_size=8, 
    stop_id_seqs: Optional[List[List[int]]] = None,
    banned_id_seqs: Optional[List[List[int]]] = None,
    banned_begin_ids: Optional[List[int]] = None,
    disable_tqdm: Optional[bool] = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0, 
    repetition_penalty: float = 1.0,  # not used
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    use_cache: bool = True,
    return_logits_for_analysis: bool = False, 
    is_instruct: bool = False,
    prompt_type: Optional[str] = None,
    **generation_kwargs,
):
    processor.tokenizer.padding_side = "left"
    if hasattr(model_wrapper, 'positive_tokenizer'):
        model_wrapper.positive_tokenizer.padding_side = "left"
    
    generations = []
    
    if not disable_tqdm:
        progress = tqdm(total=len(messages), desc="Generating")
    
    if positive_messages is None and model_wrapper.positive_model is not None:
        positive_messages = extract_text_only_messages(messages)
    
    if negative_messages is None and model_wrapper.negative_model is not None:
        negative_messages = extract_text_only_messages(messages)
    
    if positive_messages is not None:
        assert len(messages) == len(positive_messages),\
            f"messages and positive_messages should have the same length, but with {len(messages)} and {len(positive_messages)}"
    
    if negative_messages is not None: 
        assert len(messages) == len(negative_messages),\
            f"messages and negative_messages should have the same length"
        
    num_return_sequences = generation_kwargs.get('num_return_sequences', 1)
    assert num_return_sequences == 1, "num_return_sequences > 1 is not supported yet"
    
    stopping_criteria = None
    if stop_id_seqs is not None:
        stopping_criteria = StoppingCriteriaList([
            KeyWordsCriteria(stop_id_sequences=stop_id_seqs)
        ])
    
    for i in range(0, len(messages), batch_size):
        batch_messages = messages[i:i + batch_size]
        text = processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        # if prompt_type in ['vlrethinker', 'thinklite', 'r1onevision']:
        #     text = [tx if tx.endswith("<think>\n") else tx + "<think>\n" for tx in text]
        audios, images, videos = process_mm_info(batch_messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        positive_inputs = None
        negative_inputs = None
        
        if positive_messages is not None and model_wrapper.positive_model is not None:
            positive_text = [model_wrapper.positive_tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                           for msg in positive_messages[i:i + batch_size]]
            positive_text = [text if text.endswith("<think>\n") else text + "<think>\n" for text in positive_text]
            # positive_text = [text + " ... </think>\n" if text.endswith("<think>\n") else text for text in positive_text]

            positive_inputs = model_wrapper.positive_tokenizer(
                text=positive_text,
                padding=True,
                return_tensors="pt",
            ).to(model_wrapper.input_device_dict['positive'])
            
        if negative_messages is not None and model_wrapper.negative_model is not None:
            negative_text = processor.apply_chat_template(negative_messages[i:i + batch_size], tokenize=False, add_generation_prompt=True)
            # if prompt_type in ['vlrethinker', 'thinklite', 'r1onevision']:
            #     negative_text = [text if text.endswith("<think>\n") else text+ "<think>\n" for text in negative_text]
            negative_inputs = processor(
                text=negative_text,
                images=None,
                padding=True,
                return_tensors="pt",
            ).to(model_wrapper.input_device_dict['negative'])
        
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(model_wrapper.input_device_dict['base'])
        
        logit_processors = []
        if repetition_penalty != 1.0:
            logit_processors.append(
                RepetitionPenaltyLogitsProcessor(
                    penalty=repetition_penalty
                )
            )
        if banned_id_seqs is not None:
            logit_processors.append(
                NoBadWordsLogitsProcessor(
                    banned_id_seqs, eos_token_id=processor.tokenizer.eos_token_id
                )
            )
        if banned_begin_ids is not None:
            logit_processors.append(
                SuppressTokensAtBeginLogitsProcessor(
                    banned_begin_ids, begin_index=inputs.input_ids.shape[1], 
                    device=inputs.input_ids.device
                )
            )
        logits_processor = None
        if logit_processors:
            logits_processor = LogitsProcessorList(logit_processors)
            
        input_ids_len = inputs.input_ids.shape[1]
        
        batch_output_ids = model_wrapper.generate(
            inputs,
            positive_inputs, 
            negative_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            return_logits_for_analysis=return_logits_for_analysis,
            use_cache=use_cache,
            **generation_kwargs
        )
        if return_logits_for_analysis:
            batch_output_ids, analysis_data = batch_output_ids
        
        if stop_id_seqs is not None:
            for output_idx in range(batch_output_ids.shape[0]):
                for token_idx in range(input_ids_len, batch_output_ids.shape[1]):
                    if any(batch_output_ids[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_seqs):
                        batch_output_ids[output_idx, token_idx:] = processor.tokenizer.pad_token_id
                        break
                    
        batch_output_ids_trimmed = batch_output_ids[:, input_ids_len:]

        if postive_processor is not None:
            batch_output_text = postive_processor.batch_decode(
                batch_output_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        else: 
            batch_output_text = processor.batch_decode(
                batch_output_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        
        generations += batch_output_text
        
        if not disable_tqdm:
            progress.update(len(batch_messages) // num_return_sequences)

        if (i // batch_size) % 1 == 0:
            print(f"Batch {i // batch_size}: {batch_output_text[0]}")
            
    assert len(generations) == len(messages) * num_return_sequences, \
        f"Expected {len(messages) * num_return_sequences} generations, but got {len(generations)}"
        
    return generations

