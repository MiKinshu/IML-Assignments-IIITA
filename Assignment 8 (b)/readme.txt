Code by Prateek Mishra, IIT2018199 IIIT-Allahabad
In this code I have implemented facial recognition using PCA.

To get this code up and running:
1.	Install any necessary dependencies.
2.	Change the root variable to the root of the project.
3.	Make changes to the ratio, variants, imagesInEachVarient, 
	totalPixels (image width * image height) variables according
	to the dataset used.
4.	(For Imposters) Make changing to the path variable with the 
	location of the imposter image. Also ensure that the imposter
	image has the same width and length as the one defined above.
5.	Compile the project in any python compiler and observe the
	results.

Results:
	Min Accuracy is : 9.375
	Max Accuracy is : 91.875

	Imposter was also found correctly.

Dataset Description:
	There are 10 different images of 40 distinct subjects. For some of the
	subjects, the images were taken at different times, varying lighting
	slightly, facial expressions (open/closed eyes, smiling/non-smiling)
	and facial details (glasses/no-glasses).  All the images are taken
	against a dark homogeneous background and the subjects are in
	up-right, frontal position (with tolerance for some side movement).

	The files are in PGM format and can be conveniently viewed using the 'xv'
	program. The size of each image is 92x112, 8-bit grey levels. The images
	are organised in 40 directories (one for each subject) named as:

	sX

	where X indicates the subject number (between 1 and 40). In each directory
	there are 10 different images of the selected subject named as:

	Y.pgm

	where Y indicates which image for the specific subject (between 1 and 10).

About: 
	This was given to me as an assignment in the Machine Learning
	course at IIITA. I have used the ORL face dataset. I am really
	thankful for my faculty at IIITA and the authors of the dataset
	for providing such a wonderful dataset.