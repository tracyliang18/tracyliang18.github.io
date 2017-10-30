## 2017.10.17
### Done
* 协助sunlei在南北区采集攻击case
* 完成koala每日监控，图片收集
* inverse feature work log记新实验

### TODO
10%FN done
idea: total variance in grad


## 2017.10.18
* face unlock small face and dynamic face crop( smartian photo not done)
* foreground and background dataset(download done)
* sythsis dataset(sureal)  working on get-started example, can not run the game binary
* help huang to bootstrap brain++ and neupeak(done)

## 2017.10.19
* face unlock small face and dynamic face crop(setup, crop size 3.0 work much better than 2.0)
* update monitor(split file into different folders)
* koala new idea crop human(need to push sunlei to record the whole image) human shape koala
* bezier curve fitting first try

## 2017.10.20

* koala different activation function [BRIDGING NONLINEARITIES AND STOCHASTIC REGULARIZERS
WITH GAUSSIAN ERROR LINEAR UNITS](https://openreview.net/pdf?id=Bk0MRI5lg)
* face unlock traing 数据， benchmark 彻底尝试混合扣脸策略，解决大屏（纸）小脸问题
* koala 重新启动采集训练数据（pad and 纸）
* 记bezier curve fitting worklog
* 看一下STR work log

## 2017.10.21
* 动态抠脸（脸太小的抠得大一些）
* 用模型最后的抠脸测略finetune 模型，掉点！！（不科学）

## 2017.10.23
### koala
* koala 加数据
	* face id online data
	* collected
	* iphone panorama
	* 试一下加玻璃数据overfit??
* 调研koala monitor 结果的诡异之处
* koala 8bit replace_dict bug fixed, and try tanh and relu (finetune or not) experiments
* sunlei 400段活人视频 测一下 过质量判断 megface pose blur 的图
* idea: sythetic data for human action liveness
* quantize_weight [worklog](https://docs.google.com/document/d/1KV-_eZ3pYTi6B4Go8Ut52LXGafBHtrUooCS8ooyD1xs/edit)
```python
bit_width = O.shared('bit_width', np.array([8, 8]))

def quantize_w(x, num_bit=0):
    if num_bit == 0:
        num_bit = bit_width[0]

    x = O.set_grad(x, None)
    scale = O.abs(x).mean() * 2
    y = round_bit(O.clip(x / scale, -0.5, 0.5), num_bit=num_bit) * scale
    x.set_grad_var(O.grad_wrt(y))
    return y


def round_bit(x, num_bit):
    x = O.set_grad(x, None)

    max_val = 2 ** num_bit - 1
    y = (((x + 0.5) * max_val + 0.5) // 1) / max_val - 0.5

    x.set_grad_var(O.grad_wrt(y))
    return y
```
* [robotic work log](https://docs.google.com/a/megvii.com/document/d/1rzPSTBskdHLe7SEXmeqIrbJ5tWS8M-Q6rFyJbPwSjlo/edit?usp=drive_web)


### compression

npk-prune-weight
npk-cluster-weight
npk-huffman-weight

## 2017.10.24
* facepass must run at single core, current 8000+image fp 25
* train koala model using new collected data(1024)
* read deep compression
* refactor STR code,

## 10.25

## 10.26
match crop_face v2 score

## 2017.10.25
* deep compresion experiment and tool
    * alexnet and face unlock
* data list weight backprop

## 2017.10.28
* replay attack data v2
* know something about keras. and replement attack data gen using neupeak
* model_pred.py model.py 要注意



