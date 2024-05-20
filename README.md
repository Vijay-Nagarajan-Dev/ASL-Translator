# ASL-Translator
Code for video translator that translates American Sign Language letters into English, uses Kaglle mnist ASL alphabet dataset

Purpose: The purpose of the project is to translate American Sign Language to English in real time. This program will provide the user the translation of the ASL alphabet on the video panel screen. The goal of the program is to address the discrimination that people who can only communicate through ASL feel and increase communication, thus increasing productivity and as well was inclusion

Data Analysis:
Originally, I tried using the Microsoft ASL Dataset. However, this dataset’s videos were stored in YouTube, and the downloading process was strenuous. So I opted to use Kaggle’s Sign Language MNIST dataset, specifically their train dataset. 
This dataset contains 24,000 images for the letters of the alphabet in sign language. I separated the images and labels, and had to use the NumPy library to convert these images into 1D vectors, since SciKit-Learn only accepts 1D vectors for image analysis
Since the dataset has so many images for smaller amounts of information, it was bound to have repeats. This meant I had to give it much less training data so it would not end up memorizing the data, but instead learning it. This forced my train/test split to be 18%/82%, which is abnormal  

ML Model:

Then it came to model development/implementation. I utilized the SVC model from SciKit-Learn.  The kernel for the model is a radial basis function, which allows the program to deal with more complex shapes (like our hands) and deal with many decision boundaries
I used fit function to train the model based off the aforementioned Kaggle dataset (Sign Language MNIST) utilizing the .fit function
I made predictions on the testing data and using the accuracy_score() function, the accuracy of the model was 96% 


Flask + HTML + OpenCV:

First, I developed the HTML file. This is the title and the image source for the webcam using the <img> tag
I used openCV, a computer vision library, which has a function called VideoCapture(0) to open the webcam and connect it to python. 
I used the .read() function to read every frame, so that I can analyze each frame of the user using the model
Using Flask and the @app.route(‘/video_feed’), I added the webcam onto the screen.
Using the cv2.putText() function, I placed the “prediction” of the user’s ASL letter onto the camera frame.

Link For Research Paper: https://docs.google.com/document/d/1f2WDHsjvIAjYS8BfEhugvwl2V-ipgg4d3m7oUkreHMQ/edit?usp=sharing

Bibliography:

Barkved, Kirsten. "How to Know If Your Machine Learning Model Has Good Performance: Obviously Ai." Data Science without Code, 9 Mar. 2022, www.obviously.ai/post/machine-learning-model-performance#:~:text=Good%20accuracy%20in%20machine%20learning ,also%20consistent%20with%20industry%20standards. 
Saha, Avinab. "Read, Write and Display a Video Using Opencv |." LearnOpenCV, LearnOpenCV, 5 May 2021, learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/. "OpenCV - Overview." 

GeeksforGeeks, GeeksforGeeks, 26 Oct. 2022, www.geeksforgeeks.org/opencv-overview/. n/a. "Pandas Introduction." W3 Schools, Refsens Data, www.w3schools.com/python/pandas/pandas_intro.asp. 

Accessed 10 Oct. 2023. n/a. "MS-ASL." Microsoft Research, Microsoft, 5 Aug. 2019, www.microsoft.com/en-us/research/project/ms-asl/. N/a. "Scikit-Learn Machine Learning in Python." Scikit-Learn, scikit-learn.org/stable/. Accessed 10 Oct. 2023.

Cordano, Roberta J. “Is Your Organization Inclusive Of Deaf Employees?” Harvard Business Review. Harvard Business School Publishing, 11 Oct. 2022. Web. 25 Feb. 2024, https://hbr.org/2022/10/is-your-organization-inclusive-of-deaf-employeesHutchison, Graeme. “How to Build a Machine Learning Model.” Seldon, 23 Jan. 2024, www.seldon.io/how-to-build-a-machine-learning-model. I. Culjak, D. Abram, T. Pribanic, H. Dzapo and M. Cifrek, "A brief introduction to OpenCV," 2012 Proceedings of the 35th International Convention MIPRO, Opatija, Croatia, 2012, pp. 1725-1730.Abstract: The purpose of this paper is to introduce and quickly make a reader familiar with OpenCV (Open Source Computer Vision) basics without having to go through the lengthy reference manuals and books. OpenCV is an open source library for image and video analysis, originally introduced more than decade ago by Intel. Since then, a number of programmers have contributed to the most recent library developments. The latest major change took place in 2009 (OpenCV 2) which includes main changes to the C++ interface. Nowadays the library has >2500 optimized algorithms. It is extensively used around the world, having >2.5M downloads and >40K people in the user group. Regardless of whether one is a novice C++ programmer or a professional software developer, unaware of OpenCV, the main library content should be interesting for the graduate students and researchers in image processing and computer vision areas. To master every library element it is necessary to consult many books available on the topic of OpenCV. However, reading such more comprehensive material should be easier after comprehending some basics about OpenCV from this paper. keywords: {Libraries;Image edge detection;Cameras;Computer vision;Histograms;Low pass filters;Detectors},URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6240859&isnumber=6240598






