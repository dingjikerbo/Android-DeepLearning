参考这篇文章：
https://zhuanlan.zhihu.com/p/20226040

jupyter notebook --config=~/.ipython/profile_nbserver/ipython_notebook_config.py --allow-root


访问时，注意是https的
https://dingjikerbo.com:8888/

为了退出服务器时仍能用，需要nohup

按照如下文档安装gluon
http://zh.gluon.ai/chapter_preface/install.html#

另外注意如果要用mxnet，一定要先切到conda中的gluon环境，
source activate gluon

然后再nohup启动jupyter
