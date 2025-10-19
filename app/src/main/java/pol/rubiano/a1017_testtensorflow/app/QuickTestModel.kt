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
        private const val MODEL = "gpt2_spanish_quantized_dynamic.tflite"
        private const val VOCAB = "vocab.json"
    }

    fun loadModel(): Boolean {
        return try {
            val fileDescriptor = context.assets.openFd(MODEL)
            val inputStream = fileDescriptor.createInputStream()
            val mappedByteBuffer = inputStream.channel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY,
                fileDescriptor.startOffset,
                fileDescriptor.declaredLength
            )
            interpreter = Interpreter(mappedByteBuffer)

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

    private fun tokenize(prompt: String): IntArray {
        val vocab = tokenToId
        if (vocab.isEmpty()) {
            Log.e("@pol", "❌ El vocabulario (tokenToId) no está cargado.")
            return intArrayOf(50256)
        }

        val tokenIds = mutableListOf<Int>()

        val endsWithSpace = prompt.endsWith(" ")

        val words = prompt.trim().split(Regex("\\s+"))

        words.forEachIndexed { index, word ->
            if (word.isEmpty()) return@forEachIndexed

            val wordToTokenize = if (index > 0 || prompt.startsWith(" ")) "Ġ$word" else word

            var remainingText = wordToTokenize
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
                    tokenIds.add(50256)
                    remainingText = remainingText.substring(1)
                }
            }
        }

        if (endsWithSpace) {
            vocab["Ġ"]?.let { tokenIds.add(it) }
        }

        if (tokenIds.isEmpty() && prompt.isNotEmpty()) return intArrayOf(50256)
        if (tokenIds.isEmpty() && prompt.isEmpty()) tokenIds.add(vocab["<|endoftext|>"] ?: 50256)

        Log.d("@pol", "✅ Entrada: '$prompt' -> Tokens: ${tokenIds.joinToString()}")
        return tokenIds.toIntArray()
    }

    fun runInference(
        inputTokenIds: IntArray,
        generatedTokenIds: List<Int>,
        topK: Int = 40,
        temperature: Float = 0.8f,
        repetitionPenalty: Float = 1.2f
    ): Int? {
        return try {
            val interp = interpreter ?: run { Log.e("@pol", "El intérprete es nulo."); return null }

            val sequenceLength = inputTokenIds.size
            if (sequenceLength == 0) return null

            interp.resizeInput(0, intArrayOf(1, sequenceLength))
            interp.resizeInput(1, intArrayOf(1, sequenceLength))
            interp.allocateTensors()

            val inputIdsBuffer =
                ByteBuffer.allocateDirect(sequenceLength * 4).order(ByteOrder.nativeOrder())
            inputTokenIds.forEach { inputIdsBuffer.putInt(it) }
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
            val nextTokenLogits = FloatArray(vocabSize)
            System.arraycopy(allLogits, lastTokenLogitsStartIndex, nextTokenLogits, 0, vocabSize)
            for (tokenId in generatedTokenIds) {
                nextTokenLogits[tokenId] /= repetitionPenalty
            }
            for (i in nextTokenLogits.indices) {
                nextTokenLogits[i] /= temperature
            }
            val topKLogits = nextTokenLogits.mapIndexed { index, logit -> index to logit }
                .sortedByDescending { it.second }.take(topK)
            val maxLogit = topKLogits.maxOfOrNull { it.second } ?: 0.0f
            val expSum = topKLogits.sumOf { exp((it.second - maxLogit).toDouble()) }
            val probabilities =
                topKLogits.map { (exp((it.second - maxLogit).toDouble()) / expSum).toFloat() }
            val random = Random()
            val randomVal = random.nextFloat()
            var cumulativeProb = 0.0f
            var chosenIndex = -1
            for (i in probabilities.indices) {
                cumulativeProb += probabilities[i]
                if (randomVal <= cumulativeProb) {
                    chosenIndex = i; break
                }
            }
            val maxIdx =
                if (chosenIndex != -1) topKLogits[chosenIndex].first else topKLogits.firstOrNull()?.first
                    ?: 50256

            Log.d("@pol", "🎯 Token ID elegido (Top-K): id=$maxIdx")
            maxIdx

        } catch (e: Exception) {
            Log.e("@pol", "❌ Error en inferencia", e)
            null
        }
    }

    fun cleanTokenFromId(tokenId: Int): String {
        val rawToken = idToToken[tokenId] ?: "<unk>"
        val cleanToken = if (rawToken.any { it.code > 127 }) {
            try {
                String(rawToken.toByteArray(Charsets.ISO_8859_1), Charsets.UTF_8)
            } catch (e: Exception) {
                Log.e("@pol", "❌ Error limpiando token: $tokenId -> $rawToken", e)
                rawToken
            }
        } else {
            rawToken
        }
        return cleanToken.replace("?", "")
    }

    fun getTokenIdsForPrompt(prompt: String): IntArray {
        return tokenize(prompt)
    }

    fun close() {
        interpreter?.close()
    }
}
