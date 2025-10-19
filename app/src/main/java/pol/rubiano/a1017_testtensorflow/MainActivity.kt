package pol.rubiano.a1017_testtensorflow

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import pol.rubiano.a1017_testtensorflow.app.QuickTestModel

class MainActivity : AppCompatActivity() {

    private lateinit var quickTestModel: QuickTestModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        quickTestModel = QuickTestModel(this)

        if (quickTestModel.loadModel()) {
            val prompt = "Me gustaría que fuese de color"
            val result = quickTestModel.runInference(prompt)
            Log.d("@pol", "🎯 Resultado: $result")
        } else {
            Log.d("@pol", "❌ No se pudo cargar el modelo")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        quickTestModel.close()
    }
}
