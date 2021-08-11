# gesture_classification
Classify 6 different hand gestures from EMG signals (8 channels) using Bidirectional-LSTM, achiving an accuracy of 91%.

https://docs.google.com/presentation/d/1S-tZIeXz5ZrqtYjcxkbFHGBd_ZCQBcK-lrb3u8WdbEw/edit?usp=sharing

https://colab.research.google.com/drive/1ilmsd0zbhoplAQs-3-lcVRK7BD2w266t?usp=sharing

https://docs.google.com/presentation/d/1odYnBfOFzMnyk0dq3HuuZ9Qx2JZ46el6a-IdfatW9ig/edit?usp=sharing

https://docs.google.com/document/d/1fqHOCs9yaafqJTCGch0EJN-a-lbHf-e6C8YGUgfd6uU/edit?usp=sharing

https://docs.google.com/document/d/1GbC57zn7GvMN1oU75FAP7olxzZ-ef7IPO0xspEOb8HI/edit?usp=sharing

https://docs.google.com/document/d/1VmB_VKGPw4U-Pls0y5d9pWIKKpnJDRvY4_jNekYAebk/edit?usp=sharing

https://docs.google.com/document/d/1ZZsI_LtQugMpogWA_1rXggQWapcsi9YxMs8zDaq7FXk/edit?usp=sharing

## Download Data
To download data required for training/testing, follow the instruction in README (the one in raw folder)

## Before Starting
Delete README in raw folder and place holder in data folder, otherwise it will lead to error

## To Start
Download requirements
Follow documentation in main.py

## Note
The model is already trained, and main.py is set up using trained model. (the version that trains is commented)


### Evaluation Matrix
              precision    recall  f1-score   support

           1       0.99      0.98      0.99       367
           2       0.91      0.96      0.93       502
           3       0.90      0.86      0.88       523
           4       0.84      0.92      0.88       517
           5       0.90      0.84      0.87       527
           6       0.93      0.90      0.91       396

    accuracy                           0.91      2832
    macro avg      0.91      0.91      0.91      2832
    weighted avg   0.91      0.91      0.91      2832
