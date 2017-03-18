package dl4j.cawfree.github.io.networkserializationerror;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        final ByteArrayOutputStream lByteArrayOutputStream = new ByteArrayOutputStream();

        //Define a simple MultiLayerNetwork:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)
                .learningRate(0.1)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.TANH).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(3).nOut(3).build())
                .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            ModelSerializer.writeModel(net, lByteArrayOutputStream, false);

        }
        catch(IOException e) {
           e.printStackTrace();
        }

    }
}
