package pol.rubiano.a1017_testtensorflow.app

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Random
import kotlin.math.exp

class QuickTestModel(
    private val context: Context
) {
    private var interpreter: Interpreter? = null
    private lateinit var idToToken: Map<Int, String>
    private lateinit var tokenToId: Map<String, Int>

    companion object {
        private const val MODEL = "gpt2_spanish_dynamic.tflite"
        private const val VOCAB = "vocab.json"
    }

    fun loadModel(): Boolean {
        return try {
            val modelBytes = context.assets.open(MODEL).readBytes()
            val bb = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
            bb.put(modelBytes)
            bb.rewind()
            interpreter = Interpreter(bb)

            Log.d("@pol", "✅ Modelo dinámico cargado correctamente")

            loadVocab()

            Log.d("@pol", "✅ Vocabulario cargado. tamaño=${idToToken.size}")

            true
        } catch (e: Exception) {
            Log.d("@pol", "❌ Error cargando modelo: ${e.message}")
            false
        }
    }

    private fun loadVocab() {
        val reader = InputStreamReader(context.assets.open(VOCAB), "UTF-8")
        val json = JSONObject(reader.readText())
        val tempTokenToId = mutableMapOf<String, Int>()
        val tempIdToToken = mutableMapOf<Int, String>()

        json.keys().forEach { key ->
            val id = json.getInt(key)
            tempIdToToken[id] = key
            tempTokenToId[key] = id
        }

        idToToken = tempIdToToken
        tokenToId = tempTokenToId
        Log.d("@pol", "Carga de vocabulario simple y rápida completada.")
    }

    // En QuickTestModel.kt

    // En QuickTestModel.kt

    private fun tokenize(prompt: String): IntArray {
        val vocab = tokenToId
        if (vocab.isEmpty()) {
            Log.e("@pol", "❌ El vocabulario (tokenToId) no está cargado.")
            return intArrayOf(50256)
        }

        // --- REFINAMIENTO FINAL DEL TOKENIZER ---
        // Simula mejor el comportamiento del tokenizer BPE original,
        // tratando cada palabra por separado.
        val tokenIds = mutableListOf<Int>()

        // Primero, reemplazamos cualquier espacio múltiple por uno solo y quitamos espacios al inicio/final.
        val words = prompt.trim().split(Regex("\\s+"))

        words.forEachIndexed { index, word ->
            if (word.isEmpty()) return@forEachIndexed

            // Añade el prefijo de espacio 'Ġ' a todas las palabras excepto a la primera.
            val wordToTokenize = if (index > 0) "Ġ$word" else word

            var remainingText = wordToTokenize
            while (remainingText.isNotEmpty()) {
                var foundToken = false
                // Busca el sub-token más largo que coincida
                for (i in remainingText.length downTo 1) {
                    val subword = remainingText.substring(0, i)
                    if (vocab.containsKey(subword)) {
                        vocab[subword]?.let { tokenIds.add(it) }
                        remainingText = remainingText.substring(i)
                        foundToken = true
                        break
                    }
                }
                if (!foundToken) {
                    // Si no se encuentra un token, añade <unk> y avanza un carácter
                    tokenIds.add(50256)
                    remainingText = remainingText.substring(1)
                }
            }
        }

        if (tokenIds.isEmpty() && prompt.isNotEmpty()) return intArrayOf(50256)
        if (tokenIds.isEmpty() && prompt.isEmpty()) tokenIds.add(vocab["<|endoftext|>"] ?: 50256)

        Log.d("@pol", "✅ Entrada: '$prompt' -> Tokens: ${tokenIds.joinToString()}")
        return tokenIds.toIntArray()
    }



    fun runInference(    prompt: String,
                         generatedTokenIds: List<Int>, // <-- NUEVO: Pasamos los IDs ya generados
                         topK: Int = 40,
                         temperature: Float = 0.8f,
                         repetitionPenalty: Float = 1.2f
    ): String? {
        return try {
            val interp = interpreter ?: run {
                Log.e("@pol", "El intérprete es nulo.")
                return null
            }

            // 1) Tokenizar el prompt actual
            val tokenIds = tokenize(prompt)
            if (tokenIds.isEmpty()) return null
            val sequenceLength = tokenIds.size

            // 2) Redimensionar y asignar (Sin cambios)
            interp.resizeInput(0, intArrayOf(1, sequenceLength))
            interp.resizeInput(1, intArrayOf(1, sequenceLength))
            interp.allocateTensors()

            // 3) Preparar buffers de ENTRADA (Sin cambios)
            val inputIdsBuffer = ByteBuffer.allocateDirect(sequenceLength * 4).order(ByteOrder.nativeOrder())
            tokenIds.forEach { inputIdsBuffer.putInt(it) }
            inputIdsBuffer.rewind()
            val attentionMaskBuffer = ByteBuffer.allocateDirect(sequenceLength * 4).order(ByteOrder.nativeOrder())
            repeat(sequenceLength) { attentionMaskBuffer.putInt(1) }
            attentionMaskBuffer.rewind()
            val inputs = arrayOf(inputIdsBuffer, attentionMaskBuffer)

            // 4) Preparar buffer de SALIDA (Sin cambios)
            val vocabSize = 50257
            val numElements = 1 * sequenceLength * vocabSize
            val logitsBuffer = ByteBuffer.allocateDirect(numElements * 4).order(ByteOrder.nativeOrder())
            val outputs = mutableMapOf<Int, Any>()
            outputs[0] = logitsBuffer

            // 5) Ejecutar inferencia (Sin cambios)
            interp.runForMultipleInputsOutputs(inputs, outputs)

            // 6) --- PROCESAMIENTO DE SALIDA AVANZADO ---
            val logitsBb = outputs[0] as ByteBuffer
            logitsBb.rewind()
            val allLogits = FloatArray(numElements) { logitsBb.float }
            val lastTokenLogitsStartIndex = (sequenceLength - 1) * vocabSize

            // Copiar solo los logits del último token a un array propio
            val nextTokenLogits = FloatArray(vocabSize)
            System.arraycopy(allLogits, lastTokenLogitsStartIndex, nextTokenLogits, 0, vocabSize)

            // Aplicar penalización por repetición
            for (tokenId in generatedTokenIds) {
                nextTokenLogits[tokenId] /= repetitionPenalty
            }

            // Aplicar temperatura (suaviza o agudiza las probabilidades)
            for (i in nextTokenLogits.indices) {
                nextTokenLogits[i] /= temperature
            }

            // --- Top-K Sampling ---
            val topKLogits = nextTokenLogits
                .mapIndexed { index, logit -> index to logit }
                .sortedByDescending { it.second }
                .take(topK)

            // Convertir logits a probabilidades usando Softmax
            val maxLogit = topKLogits.maxOf { it.second }
            val expSum = topKLogits.sumOf { exp((it.second - maxLogit).toDouble()) }
            val probabilities = topKLogits.map {
                (exp((it.second - maxLogit).toDouble()) / expSum).toFloat()
            }

            // Muestrear (elegir) un token de los Top-K basado en sus probabilidades
            val random = Random()
            val randomVal = random.nextFloat()
            var cumulativeProb = 0.0f
            var chosenIndex = -1
            for (i in probabilities.indices) {
                cumulativeProb += probabilities[i]
                if (randomVal <= cumulativeProb) {
                    chosenIndex = i
                    break
                }
            }

            val maxIdx = if (chosenIndex != -1) topKLogits[chosenIndex].first else topKLogits.first().first

            // Devolver el token elegido
            val rawToken = idToToken[maxIdx] ?: "<unk>"

            // 2. Lo reparamos solo si es necesario
            val cleanToken = if (rawToken.any { it.code > 127 }) {
                try {
                    String(rawToken.toByteArray(Charsets.ISO_8859_1), Charsets.UTF_8)
                } catch (e: Exception) { rawToken }
            } else {
                rawToken
            }

            // 3. Lo preparamos para el usuario
            val finalToken = cleanToken
                .replace("Ġ", " ")
                .replace("?", "") // <-- ELIMINA EL SIGNO DE INTERROGACIÓN NO DESEADO

            Log.d("@pol", "🎯 Token elegido (Top-K): id=$maxIdx, raw: '$rawToken', final: '$finalToken'")
            finalToken

        } catch (e: Exception) {
            Log.e("@pol", "❌ Error en inferencia", e)
            null
        }
    }

    fun getTokenIdForToken(token: String): Int? {
        return tokenToId[token]
    }

    fun getTokenIdsForPrompt(prompt: String): IntArray {
        return tokenize(prompt)
    }

    fun close() {
        interpreter?.close()
    }
}
