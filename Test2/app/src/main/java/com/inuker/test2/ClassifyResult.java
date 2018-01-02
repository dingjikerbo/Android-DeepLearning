package com.inuker.test2;

/**
 * Created by liwentian on 2018/1/2.
 */

public class ClassifyResult {

    public float confidence;

    public String name;

    public ClassifyResult(float confidence, String name) {
        this.confidence = confidence;
        this.name = name;
    }

    @Override
    public String toString() {
        return "ClassifyResult{" +
                "confidence=" + confidence +
                ", name='" + name + '\'' +
                '}';
    }
}
