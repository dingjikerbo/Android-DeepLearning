---
title: Arm Neon资料
date: 2018-02-26 18:31:00
---

学习Neon最好的资料还是官网


包括Neon Programmar Guide，开源库



# 博客
------
[ARM NEON programming quick reference](https://community.arm.com/android-community/b/android/posts/arm-neon-programming-quick-reference#_ednref3)
https://community.arm.com/processors/b/blog/posts/coding-for-neon---part-1-load-and-stores
https://community.arm.com/processors/b/blog/posts/coding-for-neon---part-2-dealing-with-leftovers
https://community.arm.com/processors/b/blog/posts/coding-for-neon---part-3-matrix-multiplication
[NEON编码 - 第4部分: 左右移位](https://community.arm.com/cn/b/blog/posts/neon---4)
[Coding for NEON - Part 5: Rearranging Vectors](https://community.arm.com/processors/b/blog/posts/coding-for-neon---part-5-rearranging-vectors)
[ARM NEON Optimization. An Example](http://hilbert-space.de/?p=22)
http://zyddora.github.io/2016/03/16/neon_2/
https://petewarden.com/2015/10/25/an-engineers-guide-to-gemm/
https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/


# 书籍
------
1, 《并行编程方法与优化实践》http://book.51cto.com/art/201506/481002.htm

[利用NEON技术编写代码](https://community.arm.com/cn/b/blog/posts/neon)
[使用Neon优化移动设备上的C语言性能](http://ju.outofmemory.cn/entry/205929)
[ARM NEON编程初探——一个简单的BGR888转YUV444实例详解](https://segmentfault.com/a/1190000010127521)


# 开源项目
------
1，[gemmlowp](https://github.com/google/gemmlowp)
google推出的，针对uint8的gemm计算
2，[开源项目NNPACK](https://github.com/Maratyszcza/NNPACK)也是用Neon加速神经网络计算
3, https://github.com/soumith/convnet-benchmarks这个项目对比了convnet的各类方案性能
4，[Compute Library](https://community.arm.com/cn/b/blog/posts/announcing-the-compute-library-17-9-cn)
5, [mobile-deep-learning](https://github.com/baidu/mobile-deep-learning)
6, [ncnn](https://github.com/Tencent/ncnn)
7, [Caffe-HRT](https://github.com/OAID/Caffe-HRT) arm compute library在caffe上的应用
3，Ne10
4，Eigen
5, libyuv
6, skia


# 论文
------
1， [cuDNN: Efficient Primitives for Deep Learning](https://arxiv.org/pdf/1410.0759.pdf)
这篇论文论述了卷积的优化策略，包括im2col和傅里叶变换



# 其它文档
------
1，[ARM GCC Inline Assembler Cookbook](http://www.ethernut.de/en/documents/arm-inline-asm.html)
2，[ARM NEON Optimization. An Example](http://hilbert-space.de/?p=22)
3，[NEON 支持](https://developer.android.google.cn/ndk/guides/cpu-arm-neon.html?hl=zh-cn#build)
4，[Using your C compiler to exploit NEONTM Advanced SIMD](https://www.doulos.com/knowhow/arm/using_your_c_compiler_to_exploit_neon/Resources/using_your_c_compiler_to_exploit_neon.pdf)
5，[ARM C Language Extensions](http://infocenter.arm.com/help/topic/com.arm.doc.ihi0053b/IHI0053B_arm_c_language_extensions_2013.pdf)
6，[查询Neon的函数](https://developer.arm.com/technologies/neon/intrinsics)
7，[NEON](https://developer.arm.com/technologies/neon)
