package com.inuker.test2;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Created by liwentian on 2018/1/2.
 */

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
     * 大意是这和网络相关，用于将输入归一化到某个区间内
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
