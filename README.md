# Face Recognition System

This project is a complete face recognition system that performs the following tasks:

1. **Dataset Creation**
2. **Preprocessing (Embedding Generation)**
3. **Training Face Recognition Model**
4. **Recognizing Persons in Real-time**

---

## Prerequisites

Install the required libraries:
```bash
pip install opencv-python imutils numpy scikit-learn
```

## Step 1: Dataset Creation

To create a dataset for training:
```bash
python dataset_creation.py
```
### Instructions
- Enter your name and roll number when prompted.
- The system will collect 200 facial images and save them in the `dataset` folder.

---

## Step 2: Preprocessing (Embedding Generation)

To extract facial embeddings from the dataset:
```bash
python preprocessing.py
```
### Result
- `embeddings.pickle` will be generated in the `output` folder.

---

## Step 3: Train the Model

To train the SVM model for face recognition:
```bash
python train_face_ml.py
```
### Result
- `recognizer.pickle` and `le.pickle` will be created in the `output` folder.

---

## Step 4: Recognizing a Person

To recognize persons in real-time:
```bash
python recognize_person.py
```
### Instructions
- The system will recognize faces in real-time and display the person's name and roll number.
- Press `Esc` to exit the program.

---

## CSV File (`student.csv`)
- The system logs each student's name and roll number in this file.
- During recognition, this file is referenced to retrieve the corresponding roll number.

---

## Troubleshooting
- Ensure the required model files are present in the `model` folder.
- Verify your webcam is properly configured and accessible.
- For best results, ensure the camera environment has sufficient lighting.

---

## Future Improvements
- Add support for improved face detection models.
- Implement a GUI for easier usage.
- Enhance dataset handling for large-scale recognition.

---

## Contributors
- **Arivanan V.**

