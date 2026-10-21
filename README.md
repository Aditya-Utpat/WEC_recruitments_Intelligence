## profile
Name: Aditya Ajay Utpat
Roll no: 261EE103
E-mail: adityautpat07@gmail.Com

## About the project and the use of AI
While creating a machine learning model there are always two tasks, designing the architechture and implementing training and testing.

I have used AI and LLMs like chatgpt extensively for writing the code for the second part that is implementing and training.
As I am a beginner it was not feasible for me to learn to write the code for training and validation and go through so many different iterations of the model.
I will explicitly mention where so ever I used AI to write the code

# Learning
In this section, i will elaborate how i learned to create this model.
One of the main resources was 3blue 1brown's ML playlist on youtube. I had watched this playlist 2 years ago when i was in 10th. But for this project i used it as a foundational knowledge on how neural networks work, how are they trained, what are loss functions and how backpropogation is used in training.

Next the boilerplate model provided in the challenge itself acted as one of the key ways to learn convolutional neural networks. To learn CNNs I first watch the video on "what is Convolutions??" by 3blue 1 brown. Then I used chatgpt to learn how exactly the given model is working. I have linked all my chats with chatgpt so that you can see how i learned these.

I asked chatgpt to tell me about different types of layers that can be added to a CNN like this. Then i asked about what different types of loss functions can be used. I then also asked about what different type of network architechtures have been previously used like LeNet-5, AlexNet, VGGNet, GoogLeNET, ResNET. I found that being such a tiny model 


## Testing

Before even experimenting with the architechture, I built a onnxruntime script which will run on google collab notebook. I built this using ChatGPT.The script not only runs a onnx file but also tests it, meaning checks its accuracy and confusion matrix. As specified in the challenge I wanted the model on "edge hardware", I used a raspberry pi 4b 2W as my benchmark. one of the key performance matrix is RAM usage and latency time. 

In the collab NB, although you are free to look through the whole NB, I would recommend just focusing on the last block only as that is what I finally used.

## Iterations
My network architechture went over many iterations and variations

### First iteration 
First i just experimented with the number of epochs of the naive network already provided to us. Actually it helped me achieve 100% accuracy for the evaluation images just around 25-30 epochs.

### Second Iteration
I wanted to create a new network taking an inspiration from what was already provided to me. In the given architechture where i thought the network might be able to do better.

- firstly replacing the MaxPool layer with an AvgPool. My thinking was that taking only the maximum activation among four values does not capture the complete information. It might leave out contrasts which might leave out edges.
- secondly i thought that using Global Average Pool does NOT help recognize the features that each feature map wants to highlight.

So I created the following architecture, by prompting ChatGPT. You will find all prompts linked at the end of the file.

<img width="890" height="875" alt="Screenshot 2026-10-21 003111" src="https://github.com/user-attachments/assets/055b642e-eb9b-4ff2-84fe-cc164c3abb62" />
My original plan on paper
<img width="1200" height="1600" alt="WhatsApp Image 2026-10-20 at 3 07 32 PM" src="https://github.com/user-attachments/assets/a8223ef5-a54d-4fcc-8eac-074f35a59780" />

However, after training and testing this model, I found that it had low accuracy even after 40-50 epochs at which point the older more basic model seemed better.
<img width="1200" height="900" alt="confusion_matrix" src="https://github.com/user-attachments/assets/da59361f-3694-480e-b60f-0fe9b2c74cce" />

### Third Iteration

I thought that the network had become too "dense", that there were too many layers and thus it was becoming harder to train.
Also when I asked ChatGPT about the differences between AvgPool and MaxPool , It said that Maxpool helps to improve the "confidence" of the network, that is instead of the output being spread across different classes, it narrows down its selection.

### Fourth and final version

In this network I implemented the following architecture, <img width="2779" height="3592" alt="network_architecture_map_v4" src="https://github.com/user-attachments/assets/88a4ca86-6b6e-4afb-b846-48ef697fa431" />



