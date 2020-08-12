This project presents an efficient ASL recognizing system.It recognizes the ASL letters from
the ASL gestures.It also proviede text as output. We implemented both static and dynamic
gesture recognition.We get the accuracy of the ASL recognition system is 95% which is more
than the discussed reference models. This project will create an easy communication scope
between an ordinary person and who cannot speak or hear, but both have to understand
American Language and have to be able to recognize Ameican english alphabets.But the
recognition system is lightly effected due to lightening.So in the future we looking forward to
develop the model which is capable to handle lightning effect.


How to run the code?

1.creating_images.py

Create our own dataset from webcam using OpenCV by running this file.

2.rectangle.py

Here we create suares that will only shows the hand.

3.trainmodel.py

Here we train model using CNN in google colab and save the model in  .h5 format so that to use again we don't need to run all files.

4.recognize.py

Here the hand signes will be recognized based the trained model.

