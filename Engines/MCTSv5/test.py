import re


def extract_training_losses(text):
    # Use regular expressions to find training loss values
    loss_pattern = r"Training Loss: ([\d\.]+)"
    losses = re.findall(loss_pattern, text)

    # Convert the extracted losses to floats
    return [float(loss) for loss in losses]


# Example input text
text = """
Epoch 0, Training Loss: 6.4418
model saved to epoch0.pth
Training Epoch number 1
100%|██████████| 6536/6536 [23:52<00:00,  4.56it/s]
Epoch 1, Training Loss: 5.6353
model saved to epoch1.pth
Training Epoch number 2
100%|██████████| 6447/6447 [25:34<00:00,  4.20it/s]
Epoch 2, Training Loss: 5.3742
model saved to epoch2.pth
Training Epoch number 3
100%|██████████| 6572/6572 [25:12<00:00,  4.34it/s]
Epoch 3, Training Loss: 5.2039
model saved to epoch3.pth
Training Epoch number 4
100%|██████████| 6377/6377 [23:52<00:00,  4.45it/s]
Epoch 4, Training Loss: 5.0522
model saved to epoch4.pth
Training Epoch number 5
100%|██████████| 6338/6338 [22:48<00:00,  4.63it/s]
Epoch 5, Training Loss: 4.9395
model saved to epoch5.pth
Training Epoch number 6
100%|██████████| 6399/6399 [23:00<00:00,  4.63it/s]
Epoch 6, Training Loss: 4.8567
model saved to epoch6.pth
Training Epoch number 7
100%|██████████| 6475/6475 [23:14<00:00,  4.64it/s]
Epoch 7, Training Loss: 4.7869
model saved to epoch7.pth
Training Epoch number 8
100%|██████████| 6394/6394 [22:57<00:00,  4.64it/s]
Epoch 8, Training Loss: 4.7202
model saved to epoch8.pth
Training Epoch number 9
100%|██████████| 6340/6340 [22:39<00:00,  4.66it/s]
Epoch 9, Training Loss: 4.6661
model saved to epoch9.pth
Training Epoch number 10
100%|██████████| 6458/6458 [23:14<00:00,  4.63it/s]
Epoch 10, Training Loss: 4.6236
model saved to epoch10.pth
Training Epoch number 11
100%|██████████| 6341/6341 [22:38<00:00,  4.67it/s]
Epoch 11, Training Loss: 4.5900
model saved to epoch11.pth
Training Epoch number 12
100%|██████████| 6678/6678 [24:23<00:00,  4.56it/s]
Epoch 12, Training Loss: 4.5622
model saved to epoch12.pth
Training Epoch number 13
100%|██████████| 6482/6482 [23:20<00:00,  4.63it/s]
Epoch 13, Training Loss: 4.5300
model saved to epoch13.pth
Training Epoch number 14
100%|██████████| 6456/6456 [23:54<00:00,  4.50it/s]
Epoch 14, Training Loss: 4.5032
model saved to epoch14.pth
Training Epoch number 15
100%|██████████| 6615/6615 [24:13<00:00,  4.55it/s]
Epoch 15, Training Loss: 4.4661
model saved to epoch15.pth
Training Epoch number 16
100%|██████████| 6480/6480 [23:19<00:00,  4.63it/s]
Epoch 16, Training Loss: 4.4586
model saved to epoch16.pth
Training Epoch number 17
100%|██████████| 6546/6546 [23:48<00:00,  4.58it/s]
Epoch 17, Training Loss: 4.4267
model saved to epoch17.pth
Training Epoch number 18
100%|██████████| 6481/6481 [23:16<00:00,  4.64it/s]
Epoch 18, Training Loss: 4.4143
model saved to epoch18.pth
Training Epoch number 19
100%|██████████| 6569/6569 [23:52<00:00,  4.59it/s]
Epoch 19, Training Loss: 4.3851
model saved to epoch19.pth
Training Epoch number 20
100%|██████████| 6521/6521 [23:26<00:00,  4.63it/s]
Epoch 20, Training Loss: 4.3707
model saved to epoch20.pth
Training Epoch number 21
100%|██████████| 6495/6495 [23:17<00:00,  4.65it/s]
Epoch 21, Training Loss: 4.3513
model saved to epoch21.pth
Training Epoch number 22
100%|██████████| 6573/6573 [23:42<00:00,  4.62it/s]
Epoch 22, Training Loss: 4.3260
model saved to epoch22.pth
Training Epoch number 23
100%|██████████| 6445/6445 [23:15<00:00,  4.62it/s]
Epoch 23, Training Loss: 4.3412
model saved to epoch23.pth
Training Epoch number 24
100%|██████████| 6460/6460 [23:29<00:00,  4.58it/s]
Epoch 24, Training Loss: 4.3164
model saved to epoch24.pth
Training Epoch number 25
100%|██████████| 6552/6552 [24:29<00:00,  4.46it/s]
Epoch 25, Training Loss: 4.2814
model saved to epoch25.pth
Training Epoch number 26
100%|██████████| 6492/6492 [24:31<00:00,  4.41it/s]
Epoch 26, Training Loss: 4.2788
model saved to epoch26.pth
Training Epoch number 27
100%|██████████| 6455/6455 [24:27<00:00,  4.40it/s]
Epoch 27, Training Loss: 4.2545
model saved to epoch27.pth
Training Epoch number 28
100%|██████████| 6446/6446 [24:02<00:00,  4.47it/s]
Epoch 28, Training Loss: 4.2388
model saved to epoch28.pth
Training Epoch number 29
100%|██████████| 6504/6504 [24:07<00:00,  4.49it/s]
Epoch 29, Training Loss: 4.2325
model saved to epoch29.pth
Training Epoch number 30
100%|██████████| 6474/6474 [23:48<00:00,  4.53it/s]
Epoch 30, Training Loss: 4.2480
model saved to epoch30.pth
Training Epoch number 31
100%|██████████| 6408/6408 [23:48<00:00,  4.49it/s]
Epoch 31, Training Loss: 4.2156
model saved to epoch31.pth
Training Epoch number 32
100%|██████████| 6612/6612 [24:53<00:00,  4.43it/s]
Epoch 32, Training Loss: 4.2118
model saved to epoch32.pth
Training Epoch number 33
100%|██████████| 6406/6406 [23:51<00:00,  4.47it/s]
Epoch 33, Training Loss: 4.2014
model saved to epoch33.pth
Training Epoch number 34
100%|██████████| 6440/6440 [24:00<00:00,  4.47it/s]
Epoch 34, Training Loss: 4.1872
model saved to epoch34.pth
Training Epoch number 35
100%|██████████| 6402/6402 [24:06<00:00,  4.43it/s]
Epoch 35, Training Loss: 4.1816
model saved to epoch35.pth
Training Epoch number 36
100%|██████████| 6356/6356 [23:47<00:00,  4.45it/s]
Epoch 36, Training Loss: 4.1626
model saved to epoch36.pth
Training Epoch number 37
100%|██████████| 6491/6491 [24:23<00:00,  4.43it/s]
Epoch 37, Training Loss: 4.1766
model saved to epoch37.pth
Training Epoch number 38
100%|██████████| 6316/6316 [22:44<00:00,  4.63it/s]
Epoch 38, Training Loss: 4.1475
model saved to epoch38.pth
Training Epoch number 39
100%|██████████| 6515/6515 [23:41<00:00,  4.58it/s]
Epoch 39, Training Loss: 4.1564
model saved to epoch39.pth
Training Epoch number 40
100%|██████████| 6348/6348 [22:59<00:00,  4.60it/s]
Epoch 40, Training Loss: 4.1500
model saved to epoch40.pth
Training Epoch number 41
100%|██████████| 6456/6456 [23:23<00:00,  4.60it/s]
Epoch 41, Training Loss: 4.1345
model saved to epoch41.pth
Training Epoch number 42
100%|██████████| 6428/6428 [24:15<00:00,  4.42it/s]
Epoch 42, Training Loss: 4.1337
model saved to epoch42.pth
Training Epoch number 43
100%|██████████| 6425/6425 [26:25<00:00,  4.05it/s]
Epoch 43, Training Loss: 4.1147
model saved to epoch43.pth
Training Epoch number 44
100%|██████████| 6542/6542 [27:13<00:00,  4.00it/s]
Epoch 44, Training Loss: 4.1204
model saved to epoch44.pth
Training Epoch number 45
100%|██████████| 6481/6481 [26:46<00:00,  4.03it/s]
Epoch 45, Training Loss: 4.1169
model saved to epoch45.pth
Training Epoch number 46
100%|██████████| 6514/6514 [26:22<00:00,  4.12it/s]
Epoch 46, Training Loss: 4.1065
model saved to epoch46.pth
Training Epoch number 47
100%|██████████| 6507/6507 [26:50<00:00,  4.04it/s]
Epoch 47, Training Loss: 4.1071
model saved to epoch47.pth
Training Epoch number 48
100%|██████████| 6500/6500 [25:47<00:00,  4.20it/s]
Epoch 48, Training Loss: 4.0766
model saved to epoch48.pth
Training Epoch number 49
100%|██████████| 6525/6525 [25:14<00:00,  4.31it/s]
Epoch 49, Training Loss: 4.0947
model saved to epoch49.pth

100%|██████████| 6429/6429 [25:00<00:00,  4.28it/s]
Epoch 50, Training Loss: 4.0603
model saved to epoch50.pth
Training Epoch number 51
100%|██████████| 6453/6453 [25:46<00:00,  4.17it/s]
Epoch 51, Training Loss: 4.0483
model saved to epoch51.pth
Training Epoch number 52
100%|██████████| 6366/6366 [23:30<00:00,  4.51it/s]
Epoch 52, Training Loss: 4.0541
model saved to epoch52.pth
Training Epoch number 53
100%|██████████| 6532/6532 [25:20<00:00,  4.30it/s]
Epoch 53, Training Loss: 4.0546
model saved to epoch53.pth
Training Epoch number 54
100%|██████████| 6523/6523 [22:56<00:00,  4.74it/s]
Epoch 54, Training Loss: 4.0398
model saved to epoch54.pth
Training Epoch number 55
100%|██████████| 6546/6546 [23:05<00:00,  4.73it/s]
Epoch 55, Training Loss: 4.0399
model saved to epoch55.pth
Training Epoch number 56
100%|██████████| 6439/6439 [22:34<00:00,  4.75it/s]
Epoch 56, Training Loss: 4.0278
model saved to epoch56.pth
Training Epoch number 57
100%|██████████| 6525/6525 [22:55<00:00,  4.74it/s]
Epoch 57, Training Loss: 4.0311
model saved to epoch57.pth
Training Epoch number 58
100%|██████████| 6370/6370 [22:16<00:00,  4.77it/s]
Epoch 58, Training Loss: 4.0308
model saved to epoch58.pth
Training Epoch number 59
100%|██████████| 6454/6454 [22:51<00:00,  4.71it/s]
Epoch 59, Training Loss: 4.0253
model saved to epoch59.pth
Training Epoch number 60
100%|██████████| 6569/6569 [23:29<00:00,  4.66it/s]
Epoch 60, Training Loss: 4.0180
model saved to epoch60.pth
Training Epoch number 61
100%|██████████| 6370/6370 [22:29<00:00,  4.72it/s]
Epoch 61, Training Loss: 4.0139
model saved to epoch61.pth
Training Epoch number 62
100%|██████████| 6594/6594 [23:33<00:00,  4.66it/s]
Epoch 62, Training Loss: 4.0257
model saved to epoch62.pth
Training Epoch number 63
100%|██████████| 6330/6330 [22:18<00:00,  4.73it/s]
Epoch 63, Training Loss: 4.0001
model saved to epoch63.pth
Training Epoch number 64
100%|██████████| 6480/6480 [23:05<00:00,  4.68it/s]
Epoch 64, Training Loss: 4.0097
model saved to epoch64.pth
Training Epoch number 65
100%|██████████| 6454/6454 [22:43<00:00,  4.73it/s]
Epoch 65, Training Loss: 4.0023
model saved to epoch65.pth
Training Epoch number 66
100%|██████████| 6536/6536 [23:09<00:00,  4.70it/s]
Epoch 66, Training Loss: 3.9954
model saved to epoch66.pth
Training Epoch number 67
100%|██████████| 6477/6477 [22:49<00:00,  4.73it/s]
Epoch 67, Training Loss: 4.0038
model saved to epoch67.pth
Training Epoch number 68
100%|██████████| 6493/6493 [22:55<00:00,  4.72it/s]
Epoch 68, Training Loss: 3.9843
model saved to epoch68.pth
Training Epoch number 69
100%|██████████| 6518/6518 [27:22<00:00,  3.97it/s]
Epoch 69, Training Loss: 3.9875
model saved to epoch69.pth
Training Epoch number 70
100%|██████████| 6454/6454 [29:57<00:00,  3.59it/s]
Epoch 70, Training Loss: 3.9700
model saved to epoch70.pth
Training Epoch number 71
100%|██████████| 6553/6553 [26:17<00:00,  4.15it/s]
Epoch 71, Training Loss: 3.9858
model saved to epoch71.pth
Training Epoch number 72
100%|██████████| 6439/6439 [23:05<00:00,  4.65it/s]
Epoch 72, Training Loss: 3.9770
model saved to epoch72.pth
Training Epoch number 73
100%|██████████| 6429/6429 [23:20<00:00,  4.59it/s]
Epoch 73, Training Loss: 3.9589
model saved to epoch73.pth
Training Epoch number 74
100%|██████████| 6407/6407 [23:17<00:00,  4.58it/s]
Epoch 74, Training Loss: 3.9598
model saved to epoch74.pth
100%|██████████| 6464/6464 [25:01<00:00,  4.30it/s]
Epoch 75, Training Loss: 3.9498
model saved to epoch75.pth
Training Epoch number 76
100%|██████████| 6406/6406 [24:51<00:00,  4.29it/s]
Epoch 76, Training Loss: 3.9690
model saved to epoch76.pth
Training Epoch number 77
100%|██████████| 6544/6544 [24:15<00:00,  4.50it/s]
Epoch 77, Training Loss: 3.9515
model saved to epoch77.pth
Training Epoch number 78
100%|██████████| 6474/6474 [24:19<00:00,  4.43it/s]
Epoch 78, Training Loss: 3.9628
model saved to epoch78.pth
Training Epoch number 79
100%|██████████| 6499/6499 [24:46<00:00,  4.37it/s]
Epoch 79, Training Loss: 3.9477
model saved to epoch79.pth
Training Epoch number 80
100%|██████████| 6478/6478 [24:39<00:00,  4.38it/s]
Epoch 80, Training Loss: 3.9462
model saved to epoch80.pth
Training Epoch number 81
100%|██████████| 6540/6540 [26:50<00:00,  4.06it/s]
Epoch 81, Training Loss: 3.9512
model saved to epoch81.pth
Training Epoch number 82
100%|██████████| 6682/6682 [26:27<00:00,  4.21it/s]
Epoch 82, Training Loss: 3.9666
model saved to epoch82.pth
Training Epoch number 83
100%|██████████| 6446/6446 [24:46<00:00,  4.34it/s]
Epoch 83, Training Loss: 3.9397
model saved to epoch83.pth
Training Epoch number 84
100%|██████████| 6585/6585 [26:19<00:00,  4.17it/s]
Epoch 84, Training Loss: 3.9546
model saved to epoch84.pth
Training Epoch number 85
100%|██████████| 6495/6495 [26:05<00:00,  4.15it/s]
Epoch 85, Training Loss: 3.9412
model saved to epoch85.pth
"""

# Extract and print the training losses
training_losses = extract_training_losses(text)
print(training_losses)