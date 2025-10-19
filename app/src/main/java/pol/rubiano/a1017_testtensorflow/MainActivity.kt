package pol.rubiano.a1017_testtensorflow

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pol.rubiano.a1017_testtensorflow.app.QuickTestModel
import pol.rubiano.a1017_testtensorflow.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var model: QuickTestModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        CoroutineScope(Dispatchers.IO).launch {
            model = QuickTestModel(applicationContext)
            val modelLoaded = model.loadModel()
            withContext(Dispatchers.Main) {
                binding.generateButton.isEnabled = modelLoaded
                if (!modelLoaded) {
                    binding.resultTextView.text = getString(R.string.error_loading_model)
                }
            }
        }

        binding.tokenCountTextView.text = binding.tokenSlider.value.toInt().toString()

        binding.tokenSlider.addOnChangeListener { slider, value, fromUser ->
            binding.tokenCountTextView.text = value.toInt().toString()
        }

        binding.generateButton.setOnClickListener {
            val prompt = binding.promptEditText.text.toString()
            if (prompt.isNotBlank()) {
                binding.generateButton.isEnabled = false
                binding.resultTextView.text = getString(R.string.generating_text)
                val maxTokens = binding.tokenSlider.value.toInt()
                CoroutineScope(Dispatchers.Main).launch {
                    generateText(prompt, maxNewTokens = maxTokens)
                }
            }
        }
    }


    @SuppressLint("SetTextI18n")
    private suspend fun generateText(prompt: String, maxNewTokens: Int) {
        val finalGeneratedText = withContext(Dispatchers.Default) {

            val allTokenIds = model.getTokenIdsForPrompt(prompt).toMutableList()

            val generatedRawTokens = mutableListOf<String>()

            for (i in 0 until maxNewTokens) {
                Log.d("@pol", "Generando token ${i + 1}/$maxNewTokens. Total IDs: ${allTokenIds.size}")

                val nextTokenId = model.runInference(allTokenIds.toIntArray(), allTokenIds)

                if (nextTokenId == null) {
                    Log.d("@pol", "Fin de la generación: Token ID nulo.")
                    break
                }

                allTokenIds.add(nextTokenId)

                val nextRawToken = model.cleanTokenFromId(nextTokenId)

                if (nextRawToken == "<|endoftext|>") {
                    Log.d("@pol", "Fin de la generación: Token de fin de texto.")
                    break
                }
                if (nextRawToken.contains("Ċ") || nextRawToken.contains("âĢ")) {
                    Log.w("@pol", "Token basura filtrado: '$nextRawToken'. Continuando...")
                    continue
                }

                generatedRawTokens.add(nextRawToken)

                if ((nextRawToken.contains(".") || nextRawToken.contains("?") || nextRawToken.contains("!")) && generatedRawTokens.size > 3) {
                    Log.d("@pol", "Fin de la generación: Se encontró un signo de puntuación final.")
                    break
                }
            }

            val combinedText = generatedRawTokens.joinToString("")
            combinedText.replace("Ġ", " ")
        }

        binding.resultTextView.text = prompt + finalGeneratedText
        binding.generateButton.isEnabled = true
    }


    override fun onDestroy() {
        super.onDestroy()
        if (::model.isInitialized) {
            model.close()
        }
    }
}
