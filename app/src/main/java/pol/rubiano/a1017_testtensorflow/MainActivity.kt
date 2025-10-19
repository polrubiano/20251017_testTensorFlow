package pol.rubiano.a1017_testtensorflow

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

        binding.generateButton.setOnClickListener {
            val prompt = binding.promptEditText.text.toString()
            if (prompt.isNotBlank()) {
                binding.generateButton.isEnabled = false
                binding.resultTextView.text = getString(R.string.generating_text)
                CoroutineScope(Dispatchers.Main).launch {
                    generateText(prompt, maxNewTokens = 15)
                }
            }
        }
    }

    private suspend fun generateText(prompt: String, maxNewTokens: Int) {
        val generatedText = withContext(Dispatchers.Default) {
            var currentPrompt = prompt
            val allGeneratedTokens = mutableListOf<String>()
            // Obtener los IDs del prompt inicial para la primera penalización
            val allGeneratedTokenIds = model.getTokenIdsForPrompt(prompt).toMutableList()

            for (i in 0 until maxNewTokens) {
                Log.d("@pol", "Generando token ${i + 1}/$maxNewTokens. Prompt: '$currentPrompt'")

                val nextToken = model.runInference(currentPrompt, allGeneratedTokenIds)

                // --- LÓGICA DE PARADA Y FILTRADO MEJORADA ---
                if (nextToken == null || nextToken == "<|endoftext|>") {
                    Log.d("@pol", "Fin de la generación: Token nulo o de fin de texto.")
                    break
                }

                // Filtramos los tokens de control o artefactos no deseados
                if (nextToken.contains("Ċ") || nextToken.contains("âĢ") || nextToken.trim().isEmpty()) {
                    Log.w("@pol", "Token basura filtrado: '$nextToken'. Continuando...")
                    continue // Saltar al siguiente ciclo sin añadir este token
                }

                // Condición de parada si genera una frase completa
                if ( (nextToken.contains(".") || nextToken.contains("?") || nextToken.contains("!")) && allGeneratedTokens.size > 3 ) {
                    allGeneratedTokens.add(nextToken) // Añadimos el último signo de puntuación
                    Log.d("@pol", "Fin de la generación: Se encontró un signo de puntuación final.")
                    break
                }
                // --- FIN DE LA MEJORA ---

                allGeneratedTokens.add(nextToken)
                model.getTokenIdForToken(nextToken.replace(" ", "Ġ"))?.let {
                    allGeneratedTokenIds.add(it)
                }
                currentPrompt += nextToken
            }

            allGeneratedTokens.joinToString("")
        }

        binding.resultTextView.text = generatedText
        binding.generateButton.isEnabled = true
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::model.isInitialized) {
            model.close()
        }
    }
}
