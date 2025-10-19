package pol.rubiano.a1017_testtensorflow.app

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder

class QuickTestModel(
    private val context: Context
) {
    private var interpreter: Interpreter? = null
    private lateinit var idToToken: Map<Int, String>
    private lateinit var tokenToId: Map<String, Int>

    companion object {
        private const val MODEL4 = "gpt2_spanish_quantized_dynamic.tflite"
        private const val VOCAB = "vocab.json"
    }

    fun loadModel(): Boolean {
        return try {
            val modelBytes = context.assets.open(MODEL4).readBytes()
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
        val reader = InputStreamReader(context.assets.open(VOCAB))
        val json = JSONObject(reader.readText())
        tokenToId = mutableMapOf()
        idToToken = mutableMapOf()
        json.keys().forEach { key ->
            val id = json.getInt(key)
            (idToToken as MutableMap)[id] = key
            (tokenToId as MutableMap)[key] = id
        }
    }

    private fun tokenize(prompt: String): IntArray {
        val vocab = tokenToId
        if (vocab.isEmpty()) {
            Log.e("@pol", "❌ El vocabulario (tokenToId) no está cargado.")
            return intArrayOf(50256)
        }
        val textToTokenize = prompt.replace(" ", "Ġ")
        val tokenIds = mutableListOf<Int>()
        var remainingText = textToTokenize
        while (remainingText.isNotEmpty()) {
            var foundToken = false
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
                tokenIds.add(50256) // <unk>
                remainingText = remainingText.substring(1)
            }
        }
        if (tokenIds.isEmpty() && prompt.isNotEmpty()) return intArrayOf(50256)
        if (tokenIds.isEmpty() && prompt.isEmpty()) tokenIds.add(vocab["<|endoftext|>"] ?: 50256)

        Log.d("@pol", "✅ Entrada: '$prompt' -> Tokens: ${tokenIds.joinToString()}")
        return tokenIds.toIntArray()
    }

    fun runInference(prompt: String): String? {
        return try {
            val interp = interpreter ?: run {
                Log.e("@pol", "❌ El intérprete es nulo, no se puede ejecutar la inferencia.")
                return null
            }

            val tokenIds = tokenize(prompt)
            if (tokenIds.isEmpty()) return null
            val sequenceLength = tokenIds.size

            interp.resizeInput(0, intArrayOf(1, sequenceLength)) // input_ids
            interp.resizeInput(1, intArrayOf(1, sequenceLength)) // attention_mask
            interp.allocateTensors()

            val inputIdsBuffer =
                ByteBuffer.allocateDirect(sequenceLength * 4).order(ByteOrder.nativeOrder())
            tokenIds.forEach { inputIdsBuffer.putInt(it) }
            inputIdsBuffer.rewind()

            val attentionMaskBuffer =
                ByteBuffer.allocateDirect(sequenceLength * 4).order(ByteOrder.nativeOrder())
            repeat(sequenceLength) { attentionMaskBuffer.putInt(1) }
            attentionMaskBuffer.rewind()

            val inputs = arrayOf(inputIdsBuffer, attentionMaskBuffer)

            val vocabSize = 50257
            val numElements = 1 * sequenceLength * vocabSize
            val logitsBuffer =
                ByteBuffer.allocateDirect(numElements * 4).order(ByteOrder.nativeOrder())

            val outputs = mutableMapOf<Int, Any>()
            outputs[0] = logitsBuffer

            interp.runForMultipleInputsOutputs(inputs, outputs)

            val logitsBb = outputs[0] as ByteBuffer
            logitsBb.rewind()

            val allLogits = FloatArray(numElements) { logitsBb.float }
            val lastTokenLogitsStartIndex = (sequenceLength - 1) * vocabSize

            var maxIdx = -1
            var maxLogit = -Float.MAX_VALUE
            for (i in 0 until vocabSize) {
                val currentLogit = allLogits[lastTokenLogitsStartIndex + i]
                if (currentLogit > maxLogit) {
                    maxLogit = currentLogit
                    maxIdx = i
                }
            }
            if (maxIdx == -1) maxIdx = 50256

            val token = idToToken[maxIdx] ?: "<unk>"
            val finalToken = token.replace("Ġ", " ")
            Log.d("@pol", "✅ Token elegido id=$maxIdx token='$token' score=$maxLogit")

            finalToken

        } catch (e: Exception) {
            Log.e("@pol", "❌ Error en inferencia", e)
            null
        }
    }


    fun close() {
        interpreter?.close()
    }
}
