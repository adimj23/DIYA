# DIYA

# Project: The Effect of Controllable Features on Acceptance of DonorsChoose Applications

 Adi Srinivasan, 7-14-20


## _Come up with a name for your summer project. _ \
 \
Describe the Question 

_Describe the question that you chose from last week. Link it to your project plan here.  _

The question I have chosen to pursue is to see if we can predict if an application will be accepted based on the aspects of the application that are within control of the teacher.


## Dataset Exploration 

_Identify the variable to be predicted (target variable) and identify the features that will be used for prediction. Provide some descriptive statistics of the variables you intend to use for the prediction (classification or regression). How would you justify the inclusion of the set of features to predict the target variable?_

The target variable would be the one indicating whether or not the project was approved. This would be a classification model, sorting instances into groups of was approved or wasn’t approved.

The features used for prediction will be the month of submission, the length of project subcategory, the length of the project title, frequency of keywords in the essays, the length of the essays, and the length of resource summary.

I am not including the project category because teachers have less discretion over choosing that, as there are a limited number of options they can choose from. While they can’t choose what subject they teach, they can choose the wording of their subcategory, which is why I am including subcategory as a controllable feature.

These are all features I am choosing to include because teachers have some control over all of them, and they are all given data points that could show some correlation with a higher acceptance rate. 


## Data Cleaning/Manipulation 

_Figure out if the data can be used as is. Are all the records complete with the values you would use for training? Do you need to eliminate certain records? Do you need to consolidate or convert any fields? _

All applications after 2016-05-17 will have missing entries for essays 3 and 4. I don’t think I’ll need to eliminate those records, I could either separate them or leave them as is and only predict based on the essays that are available. 


## Training Set/Test Set

_Determine how you will divide the data into a training set, validation set and test set. _

I will use an 80-10-10 ratio of splitting my data into training, validation, and test sets respectively.


## Machine Learning Algorithm 

_Explore the use of the decision tree algorithm for solving the problem. Describe the decision tree used for solving the problem. Add code snippets to showcase the approach that you took to solve the problem. _

_Add any assumptions that you are making on the dataset while applying the machine learning algorithm. _

I used a decision tree classifier for my first algorithm. My control features ended up being length of the essays, month of submission, length of project title, length of resource summary, and the frequency of the top 50 keywords of approved essays based on TF-IDF scores.



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")


The above code snippet shows the use of the TfidfTransformer from sklearn, which I used to find the top 50 keywords used in all of the approved essays. I decided to consolidate all essays into a single column.  



<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")



## Performance Evaluation and Analyses

_Evaluate the performance of the “decision tree” algorithm to answer your question by calculating the accuracy of the algorithm on the training data and validation data. Think of setting up a realistic experimental condition, where you carefully observe the changes in performance by varying different factors in a controlled manner. To do that, observe how the performance of the algorithm changes with respect to:_



1. _Size of the training data. To see this plot the training accuracy and validation accuracy for classification (mean squared error for regression) for different values of training data size. What are your observations?_

    

<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image3.png "image_tooltip")



    

<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.png "image_tooltip")



    

<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.png "image_tooltip")


2. _Parameters of the decision tree algorithm. To see this, plot the training accuracy and validation accuracy for (i) different values of the minimum number of samples required at a leaf node (ii) different values of the maximum depth of the tree._

_Remember to avail the benefits of various visualization tools to aid your analyses._

Min Sample Leaf



<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image6.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image6.png "image_tooltip")




<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image7.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image7.png "image_tooltip")


Not sure if there was an error on my part but the min_samples_leaf doesn’t seem to affect accuracy

Max Depth



<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.png "image_tooltip")




<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image9.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image9.png "image_tooltip")



## Analyses of Results

_Share your analyses of the results. To help you with this, evaluate the performance by checking for overfitting or underfitting. _How would you address the overfitting/underfitting conditions?

What is the best setting of the decision tree? Plot your tree.



<p id="gdcalert10" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image10.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert11">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image10.png "image_tooltip")



## Interpret your Model

Explain the features that were useful in the prediction. Are some variables more important than others in your prediction? Are you surprised by what you found?
