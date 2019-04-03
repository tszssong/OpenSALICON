#### OpenSalicon 主干网络改为 resnet 18
###### 脚本说明  
    1. testDirTxtOriOut.py list.txt
        为x264编码提供50x38的显著图与对应txt, list.txt保存视频名
        
###### 训练  
    finetune_salicon_res.py  预训练模型在脚本内部指定 
    所用协议finetune_salicon_res.prototxt, solver_mew.prototxt

###### 测试  
    SaliconRes.py为forwad模型脚本，所用协议salicon_res18.prototxt
    1. testPic.py
        测试工程路径下'face.jpg'对应显著图
    2. testPicDir.py 
        批量测试静态图显著性，test.txt保存测试图片名
