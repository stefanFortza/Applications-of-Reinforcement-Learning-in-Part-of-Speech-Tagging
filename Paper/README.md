Note: totaly robica generated content, unfortunetly :<

# Contains all the related materials needed for writing the final document

## Title

## Authors

## Abstract
This should be an overview about the final paper, i recommand this being the last one wrote with the conclusion, just to have an accurate description

Mostlikely it should outline the fact that the reinforcement learning was not aborted as traditional tehnique in POS tagging literature and we wanted to see if it is helpfull or not in POS tagging related tasks.

## Introduction

Short description about the research question we tried to investigate:

RQ1: How well will perform a RL POS tagger method compared to the current baselines on the english language?

RQ2: Can an RL Agent trained to use a POS tagger or ask for an oracle perform well?

Here it is describe the scope of pos tagging why it is usefull and so on, POS tags being one of the fundamental features used in NLP for different task, predominante being ... (maybe authorship detection, maybe sentimant analassys, etc.) Another important thing must to mention is the posible POS tags and that our research is focused on the Universal Tags. Explaining why the idea of impling RL could had potential, detailing the RQ2, when some contexts creates so hard to categorize some POS so that it is better to flag them and to be analyzed by a human being.

Describing the structure of the paper in parts maybe

## Related Work

Here should be search related research work on Offline RL, Similar pos tagging tasks using an oracle.

This should be made extremly correct, this will tell if this project is good one or a weekend crap ...

Aaa yeah, the state of the art approach in POS tagging, most likely bidirectional LSTM, and some references about the baseline model which will be used :/

## Part 1

this cover the research question 1, firstly let s define the baseline model, most likely a markov statistical model and the corpus (dataset) used for the training, than the description of GYM enviroment used for the experiment. Explain that we used a DQN, than will follow with a discousin about the reward function and the state representation, do not forget to talk about the model travers the sentence left to right, so it is missing a lot of the arouding context. And by the end tested the best model compared to the baseline, most likely it will fail but it can be mentioned that the model learned to generalize. Mention that the main cause of failure being like the global reward, the model even if it got a bad reward could not realize which exact word was the problem.

## Part 2

this cover the research question 2, as before defines the enviroment, talk how the helper pos tagger was trained and talk about different types of learning used, DQN, Q learing, Double dqn, rainbow dqn. Compare them and state the best one of them. Now being defined the model benchmark it in POS tagging and compare to the baseline and to only applying the helper pos tagger, see if the accuracy increase by using the oracle in most likely hard word to be tagged. than the last accuracy test to be the comparation when the action is to call the oracle, if the pos tagger would made a mistake or not, to see if it actually make good action to only remain hard to identify words.

## Conclusion

This concludes the results, most likely the fail of question 1 and second of all the results of question 2 which more likely will be more usefull ones, 

## Future work

here to mention possible new direction to research, maybe testing on the other types of pos tags, or applying that oracle method on other nlp tasks, something concise

## Bibligraphy

cite some papers


