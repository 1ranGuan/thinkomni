# ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding (ICLR 2026)

ThinkOmni is a **training-free** framework that enhances **omni-modal LLMs (OLLMs)** with the **reasoning ability of large reasoning models (LRMs)** via **guidance decoding**.  
Instead of additional finetuning, ThinkOmni integrates an off-the-shelf LRM at **decoding time** and adaptively balances perception vs. reasoning signals for robust multi-modal reasoning.

- **arXiv**: https://arxiv.org/abs/2602.23306

## Highlights

- **Training-free omni-modal reasoning boost**: no SFT/RFT required.
- **LRM-as-a-Guide**: uses an off-the-shelf reasoning LLM to guide OLLM decoding.
- **Stepwise Contrastive Scaling (SCS)**: automatically adjusts guidance strength step-by-step.

<!-- Full code will be open-sourced under the **[xiaomi-research](https://github.com/xiaomi-research)**. Please stay tuned! -->

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{guan2026thinkomni,
  title={ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding},
  author={Guan, Yiran and Tu, Sifan and Liang, Dingkang and Zhu, Linghao and Ju, Jianzhong and Luo, Zhenbo and Luan, Jian and Liu, Yuliang and Bai, Xiang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}