package com.inuker.test1;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends Activity {

    private static final String MODEL_FILE = "file:///android_asset/optimized_tfdroid.pb";
    private static final String INPUT_NODE = "X";
    private static final String OUTPUT_NODE = "Y";

    private static final long[] INPUT_SIZE = {1,3};

    private EditText mInput1;
    private EditText mInput2;
    private EditText mInput3;

    private TensorFlowInferenceInterface inferenceInterface;

    private Button mBtn;

    private TextView mTextView;

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

        mTextView = findViewById(R.id.out);

        mBtn = findViewById(R.id.btn);

        mBtn.setOnClickListener(new View.OnClickListener() {

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

        float[] resu = {0, 0};
        inferenceInterface.fetch(OUTPUT_NODE, resu);

        mTextView.setText(resu[0] + ", " + resu[1]);
    }
}
