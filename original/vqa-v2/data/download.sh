#!/bin/sh

# questions
curl -O  https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip >/dev/null

# answers
curl -O  https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip >/dev/null

# balanced pairs
curl -O  https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip >/dev/null

# bottom up features (https://github.com/peteanderson80/bottom-up-attention)
curl -O  https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip https://imagecaption.blob.core.windows.net/imagecaption/test2015.zip >/dev/null
## alternative bottom-up features: 36 fixed proposals per image instead of 10--100 adaptive proposals per image.
#curl -O https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip https://imagecaption.blob.core.windows.net/imagecaption/test2015_36.zip

unzip "*.zip"
