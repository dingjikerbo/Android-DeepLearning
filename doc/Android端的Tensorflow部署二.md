---
title: 写给Android程序员的Tensorflow教程二
date: 2018-01-02 19:31:25
---

本文将展示一个稍微复杂点的例子，仍然不涉及模型训练，只是导入一个别人已经训好的模型，运行看效果。这个模型是用于物体分类的，打开相机拍照，识别图片中物体并给出识别结果及相应概率。

模型下载路径
https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip

解压后有三个文件：

```
imagenet_comp_graph_label_strings.txt
LICENSE
tensorflow_inception_graph.pb
```

创建一个App工程，将imagenet_comp_graph_label_strings.txt和tensorflow_inception_graph.pb拷到assets目录，导入tensorflow需要的jar和so库文件，

接下来为了方便使用相机，导入一个相机开源库，

```
compile 'com.flurgle:camerakit:0.9.12'
```

然后定义一个ImageClassifier类，用于图片分类，如下，

```
public class ImageClassifier {

    private TensorFlowInferenceInterface mTensorflow;
    private static final float THRESHOLD = 0.1f;
    private static final int MAX_RESULTS = 3;

    private String mInputName;
    private String mOutputName;
    private String[] mOutputNames;
    private int[] mIntValues;
    private float[] mFloatValues;

    /**
     * 输入图片是InputSize * InputSize * 3的
     */
    private int mInputSize;

    /**
     * 关于IMAGE_MEAN和IMAGE_STD的解释可参考
     * https://github.com/googlecodelabs/tensorflow-for-poets-2/issues/2
     * https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c
     * 大意是这和网络相关，用于将输入标准化到某个区间内
     */
    private int mImageMean;
    private float mImageStd;

    private float[] mOutputs;

    private List<String> mLabels = new ArrayList<>();

    public ImageClassifier(AssetManager assetManager, String modelFilename, String labelFilename,
            int inputSize, int imageMean, float imageStd,
            String inputName, String outputName) {
        mInputName = inputName;
        mOutputName = outputName;

        readLabelFile(assetManager, labelFilename);

        mTensorflow = new TensorFlowInferenceInterface(assetManager, modelFilename);

        mInputSize = inputSize;
        mImageMean = imageMean;
        mImageStd = imageStd;

        mOutputNames = new String[]{outputName};
        mIntValues = new int[inputSize * inputSize];
        mFloatValues = new float[inputSize * inputSize * 3];

        // numClasses为输出结果的个数，每个结果对应一个概率
        int numClasses = (int) mTensorflow.graph().operation(outputName).output(0).shape().size(1);
        mOutputs = new float[numClasses];
    }

    /**
     * 读取label文件
     */
    private void readLabelFile(AssetManager assetManager, String labelFilename) {
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                mLabels.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public List<ClassifyResult> recognizeImage(final Bitmap bitmap) {
        bitmap.getPixels(mIntValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < mIntValues.length; ++i) {
            final int val = mIntValues[i];
            // 对图片进行标准化处理
            mFloatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - mImageMean) / mImageStd;
            mFloatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - mImageMean) / mImageStd;
            mFloatValues[i * 3 + 2] = ((val & 0xFF) - mImageMean) / mImageStd;
        }
        mTensorflow.feed(mInputName, mFloatValues, new long[] {1, mInputSize, mInputSize, 3});
        mTensorflow.run(mOutputNames, true);
        mTensorflow.fetch(mOutputName, mOutputs);

        mQueue.clear();
        for (int i = 0; i < mOutputs.length; i++) {
            if (mOutputs[i] > THRESHOLD) {
                mQueue.add(new ClassifyResult(mOutputs[i], mLabels.get(i)));
            }
            if (mQueue.size() > MAX_RESULTS) {
                mQueue.poll();
            }
        }

        List<ClassifyResult> results = new ArrayList<>();
        while (!mQueue.isEmpty()) {
            results.add(0, mQueue.poll());
        }

        return results;
    }

    private final PriorityQueue<ClassifyResult> mQueue = new PriorityQueue<>(3, new Comparator<ClassifyResult>() {

        @Override
        public int compare(ClassifyResult o1, ClassifyResult o2) {
            return Float.compare(o1.confidence, o2.confidence);
        }
    });

    public void close() {
        mTensorflow.close();
    }
}
```

这里要读取label文件，便于之后根据output的index取得对应的label，output的是对应的概率。这里用一个优先队列保存概率最大的三个物体返回。

最后再来创建Activity，CameraView用于相机预览，当点击按钮时抓拍并将图片数据丢给ImageClassifier获得识别结果，然后显示出来。

```
public class MainActivity extends Activity {

    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";

    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt";

    private CameraView mCamera;

    private ImageClassifier mClassifier;

    private TextView mTvResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mClassifier = new ImageClassifier(getAssets(), MODEL_FILE, LABEL_FILE, INPUT_SIZE, IMAGE_MEAN, IMAGE_STD, INPUT_NAME, OUTPUT_NAME);

        mTvResult = findViewById(R.id.result);
        mCamera = findViewById(R.id.camera);
        mCamera.setCameraListener(new CameraListener() {
            @Override
            public void onPictureTaken(byte[] jpeg) {
                Bitmap bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
                List<ClassifyResult> results = mClassifier.recognizeImage(bitmap);
                mTvResult.setText(results.toString());
            }
        });

        findViewById(R.id.detect).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mCamera.captureImage();
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        mCamera.start();
    }

    @Override
    protected void onPause() {
        mCamera.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        mClassifier.close();
        super.onDestroy();
    }
}
```

这个项目可以和tensorflow自带的Android例子对照着看，基本是相似的。参考tensorflow/examples/android中的ClassifierActivity.java和TensorFlowImageClassifier.java。

不过奇怪的是识别的结果中概率都很低，普遍是20%左右，原因不明。
下文将研究如何在已训好的模型基础上增量训练我们自己的数据。

另外可以参照poets，
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#1
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/index.html#0
https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/

[Android TensorFlow Machine Learning Example](https://blog.mindorks.com/android-tensorflow-machine-learning-example-ff0e9b2654cc)

[MindorksOpenSource/AndroidTensorFlowMachineLearningExample](https://github.com/MindorksOpenSource/AndroidTensorFlowMachineLearningExample)


