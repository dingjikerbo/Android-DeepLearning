本文将展示如何将一个简单的tensorflow模型加载到Android app，并能看到运行效果。这里的模型是个简单的线性函数，没有训练过程，直接给参数赋值。执行如下python程序，只是给定义的网络保存到文件中了。

```
import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None, 3], name='X')  # input
W = tf.Variable(tf.zeros(shape=[3, 2]), dtype=tf.float32, name='W')  # weights
b = tf.Variable(tf.zeros(shape=[2]), dtype=tf.float32, name='b')  # biases
Y = tf.nn.relu(tf.matmul(X, W) + b, name='Y')  # activation / output

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # save the graph
    tf.train.write_graph(sess.graph_def, '.', 'tfdroid.pbtxt')

    # normally you would do some training here
    # but fornow we will just assign something to W
    sess.run(tf.assign(W, [[1, 2], [4, 5], [7, 8]]))
    sess.run(tf.assign(b, [1, 1]))

    # save a checkpoint file, which will store the above assignment
    saver.save(sess, './tfdroid.ckpt')
```

注意最后这个saver.save里的路径别掉了"./"，否则会提示找不到父目录。运行会生成如下文件：

```
tfdroid.ckpt.index
tfdroid.ckpt.meta
tfdroid.pbtxt
```

其中pbtxt文件是计算图的定义，ckpt文件是各种模型参数。

接下来要将这两个文件合二为一，并做一些优化减少最终文件的体积。执行如下python程序，这里freeze_graph.py和optimize_for_inference_lib.py都在tensorflow/python/tools目录下，

```
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import tensorflow as tf

MODEL_NAME = 'tfdroid'

# Freeze the graph

input_graph_path = MODEL_NAME+'.pbtxt'
checkpoint_path = './'+MODEL_NAME+'.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "Y"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "r") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["X"], # an array of the input node(s)
        ["Y"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph

f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())
```

注意输入节点名称是X，输出结点名称是Y，这里面别搞错了。最后生成了如下两个文件，

```
frozen_tfdroid.pb
optimized_tfdroid.pb
```

我们要用的就是这个optimized_tfdroid.pb文件了，接下来在Android Studio中创建一个App工程，导入tensorflow的jar和so文件，[下载链接](https://ci.tensorflow.org/view/Nightly/job/nightly-android/)，要导入如下两个文件，

```
libtensorflow_inference.so
libandroid_tensorflow_inference_java.jar
```

这两个文件都放到libs目录下，在build.gradle的android block中添加

```
sourceSets {
    main {
        jniLibs.srcDirs = ['libs']
    }
}
```

然后给optimized_tfdroid.pb文件放到assets目录下，接下来定义Activity，

```
public class MainActivity extends Activity {

    private static final String MODEL_FILE = "file:///android_asset/optimized_tfdroid.pb";
    private static final String INPUT_NODE = "X";
    private static final String OUTPUT_NODE = "Y";

    private static final long[] INPUT_SIZE = {1,3};

    private EditText mInput1;
    private EditText mInput2;
    private EditText mInput3;

    private TensorFlowInferenceInterface inferenceInterface;

    private Button mBtnRun;

    private TextView mTvResult;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mInput1 = findViewById(R.id.input1);
        mInput2 = findViewById(R.id.input2);
        mInput3 = findViewById(R.id.input3);

        mTvResult = findViewById(R.id.out);

        mBtnRun = findViewById(R.id.btn);

        mBtnRun.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                test();
            }
        });

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    }

    private void test() {
        float num1 = Float.valueOf(mInput1.getText().toString().trim());
        float num2 = Float.valueOf(mInput2.getText().toString().trim());
        float num3 = Float.valueOf(mInput3.getText().toString().trim());

        float[] inputFloats = {num1, num2, num3};

        inferenceInterface.feed(INPUT_NODE, inputFloats, INPUT_SIZE);
        inferenceInterface.run(new String[] {OUTPUT_NODE}, true);

        float[] result = {0, 0};
        inferenceInterface.fetch(OUTPUT_NODE, result);

        mTvResult.setText(resu[0] + ", " + resu[1]);
    }
}
```

因为我们定义的网络的输入是3列的，输出是2列的，所以这里输入了三个浮点，输出传入了一个float[2]。

运行的结果符合预期，就是个简单的矩阵乘法。

Demo下载链接 - [Android-DeepLearning](https://github.com/dingjikerbo/Android-DeepLearning/tree/master/test1)

这篇文章的意义就是让我们了解如何将tensorflow设计好的网络在Android上跑通。正常的流程应该是用tensorflow定义好网络，然后输入数据进行训练，训练的结果就是各个模型参数的值，保存优化成pb文件，然后Android中导入运行即可。我们这里省略了模型训练部分，之后的文章中会逐步展开。

可参考如下文章
1. [Deploying a TensorFlow model to Android](https://medium.com/joytunes/deploying-a-tensorflow-model-to-android-69d04d1b0cba)
2. [Tutorial: Build Your First Tensorflow Android App](https://omid.al/posts/2017-02-20-Tutorial-Build-Your-First-Tensorflow-Android-App.html)

