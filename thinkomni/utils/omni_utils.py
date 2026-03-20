def get_audio_score_ICE():
    example_1 = """
[Question]: What was the main topic discussed in the audio clip?
Choices:
A: Budget allocation
B: Project timeline
C: Team reorganization
D: Client feedback
[Standard Answer]: B
[Model_answer]: Extracted Answer: B
Judgement: 1
""" # noqa

    example_2 = """
[Question]: According to the speaker, when is the deadline for the report?
Choices:
A: Monday
B: Wednesday
C: Friday
D: Next week
[Standard Answer]: C
[Model_answer]: Extracted Answer: B
Judgement: 0
""" # noqa

    example_3 = """
[Question]: What emotion did the speaker primarily express?
Choices:
A: Excitement
B: Frustration
C: Indifference
D: Confusion
[Standard Answer]: B
[Model_answer]: Extracted Answer: null
Judgement: 0
""" # noqa

    example_4 = """
[Question]: Where does the speaker suggest holding the next meeting?
Choices:
A: Conference room
B: Coffee shop
C: Virtual platform
D: Client's office
[Standard Answer]: C
[Model_answer]: Extracted Answer: C
Judgement: 1
""" # noqa

    example_5 = """
[Question]: What was the speaker's main recommendation?
Choices:
A: Hire more staff
B: Extend the deadline
C: Reduce project scope
D: Increase the budget
[Standard Answer]: B
[Model_answer]: Extracted Answer: B. Extend the deadline
Judgement: 1
""" # noqa

    example_6 = """
[Question]: How many participants were mentioned in the meeting?
Choices:
A: 5
B: 7
C: 9
D: 11
[Standard Answer]: B
[Model_answer]: Extracted Answer: seven
Judgement: 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]

def get_audio_extract_ICE():
    example_1 = """
1.
Model response: 'Based on the audio content, the speaker's tone suggests the correct answer is option B.'
Extracted Answer: B
""" # noqa

    example_2 = """
2.
Model response: 'After analyzing the speech patterns and background sounds, the most likely answer is:\n\nC. The meeting was postponed'
Extracted Answer: C
""" # noqa

    example_3 = """
3.
Model response: 'The audio clip clearly indicates that the correct choice is A, as the speaker explicitly mentions "the first option is correct".'
Extracted Answer: A
""" # noqa

    example_4 = """
4.
Model response: 'From the emotional tone and speech content, the answer should be D, which matches the speaker's implied meaning.'
Extracted Answer: D
""" # noqa

    example_5 = """
5.
Model response: 'The audio quality is too poor to determine the correct answer with confidence.'
Extracted Answer: null
""" # noqa

    example_6 = """
6.
Model response: 'After careful consideration of both the verbal content and non-verbal cues in the recording, the correct answer is:\n\nB. The project deadline has been extended'
Extracted Answer: B
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]

def get_vision_extract_ICE():
    example_1 = """
1.
Model response: 'The image shows a red traffic light, so the correct answer is:\n\nB. Stop'
Extracted Answer: B
""" # noqa

    example_2 = """
2.
Model response: 'Based on the objects in the image, the most likely scene is:\n\nD. A kitchen with a refrigerator and stove.'
Extracted Answer: D
""" # noqa

    example_3 = """
3.
Model response: 'The text in the image reads "EXIT", which matches option A.'
Extracted Answer: A
""" # noqa

    example_4 = """
4.
Model response: 'The image is too blurry to identify the number of people present.'
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: 'The dominant color in the image is blue, corresponding to option C.'
Extracted Answer: C
""" # noqa

    example_6 = """
6.
Model response: 'The animal in the image has stripes, confirming the answer is:\n\nB. Zebra'
Extracted Answer: B
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]

def get_vision_score_ICE():
    example_1 = """
[Question]: What is the primary color of the object in the image?
Choices:
A: Red
B: Green
C: Blue
D: Yellow
[Standard Answer]: A
[Model_answer]: Extracted Answer: A
Judgement: 1
""" # noqa

    example_2 = """
[Question]: How many people are visible in the image?
Choices:
A: 1
B: 2
C: 3
D: 4
[Standard Answer]: B
[Model_answer]: Extracted Answer: C
Judgement: 0
""" # noqa

    example_3 = """
[Question]: What is the text written on the sign in the image?
Choices:
A: "OPEN"
B: "CLOSED"
C: "PUSH"
D: "PULL"
[Standard Answer]: D
[Model_answer]: Extracted Answer: D. "PULL"
Judgement: 1
""" # noqa

    example_4 = """
[Question]: What type of animal is shown in the image?
Choices:
A: Cat
B: Dog
C: Rabbit
D: Bird
[Standard Answer]: B
[Model_answer]: Extracted Answer: null
Judgement: 0
""" # noqa

    example_5 = """
[Question]: What is the main activity happening in the image?
Choices:
A: Cooking
B: Dancing
C: Reading
D: Sleeping
[Standard Answer]: A
[Model_answer]: Extracted Answer: A (a person chopping vegetables)
Judgement: 1
""" # noqa

    example_6 = """
[Question]: What is the shape of the object in the center of the image?
Choices:
A: Circle
B: Square
C: Triangle
D: Star
[Standard Answer]: C
[Model_answer]: Extracted Answer: Triangle
Judgement: 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]

def get_multimodal_extract_ICE():
    example_1 = """
1.
Model response: 'The image shows a sunny beach, and the audio mentions "vacation plans", so the correct answer is:\n\nA. Travel'
Extracted Answer: A
""" # noqa

    example_2 = """
2.
Model response: 'Combining the image of a broken window and the audio of a scream, the most likely event is:\n\nC. An accident'
Extracted Answer: C
""" # noqa

    example_3 = """
3.
Model response: 'The audio says "meeting at 3 PM", and the image shows a clock pointing to 3:00, confirming option B.'
Extracted Answer: B
""" # noqa

    example_4 = """
4.
Model response: 'The image is a blank screen, and the audio is silent. No answer can be determined.'
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: 'The audio describes "a red bird singing", while the image shows a cardinal. The answer is:\n\nD. Cardinal'
Extracted Answer: D
""" # noqa

    example_6 = """
6.
Model response: 'The image shows a crowded street, and the audio says "protest". The best match is:\n\nB. Demonstration'
Extracted Answer: B
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]

def get_multimodal_score_ICE():
    example_1 = """
[Question]: What is the combined context of the image and audio?
Choices:
A: Travel
B: Work
C: Study
D: Exercise
[Standard Answer]: A
[Model_answer]: Extracted Answer: A
Judgement: 1
""" # noqa

    example_2 = """
[Question]: What emotion is conveyed by the image and audio together?
Choices:
A: Joy
B: Fear
C: Anger
D: Sadness
[Standard Answer]: B
[Model_answer]: Extracted Answer: C
Judgement: 0
""" # noqa

    example_3 = """
[Question]: What time is indicated by the image (clock) and audio ("meet at 3 PM")?
Choices:
A: 1 PM
B: 3 PM
C: 5 PM
D: 7 PM
[Standard Answer]: B
[Model_answer]: Extracted Answer: B
Judgement: 1
""" # noqa

    example_4 = """
[Question]: What is the main object shown in the image and described in the audio?
Choices:
A: Dog
B: Cat
C: Bird
D: Fish
[Standard Answer]: C
[Model_answer]: Extracted Answer: null
Judgement: 0
""" # noqa

    example_5 = """
[Question]: What action is suggested by the image (person running) and audio ("quick, hurry!")?
Choices:
A: Walking
B: Running
C: Sleeping
D: Eating
[Standard Answer]: B
[Model_answer]: Extracted Answer: B (running)
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Where does the scene likely take place based on the image (snow) and audio ("freezing")?
Choices:
A: Desert
B: Forest
C: Arctic
D: Ocean
[Standard Answer]: C
[Model_answer]: Extracted Answer: Arctic
Judgement: 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]