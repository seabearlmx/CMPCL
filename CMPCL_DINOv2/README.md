Other settings follow [Rein](https://github.com/w1oves/Rein)
## Training
```
PORT=12345 CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/dinov2/dinov2_mask2former_512x512_bs1x4.py 4
```

## Evaluation
  Run the evaluation:
  ```
  python tools/test.py configs/dinov2/dinov2_mask2former_512x512_bs1x4.py work_dirs/dinov2_mask2former_512x512_bs1x4/iter_40000.pth
  ```

## Acknowledgment
Our implementation is mainly based on the following repositories. Thanks to their authors.
* [Rein](https://github.com/w1oves/Rein)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
