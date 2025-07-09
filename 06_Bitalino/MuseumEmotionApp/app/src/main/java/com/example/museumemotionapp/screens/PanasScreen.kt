package com.example.museumemotionapp.screens

import android.content.Context
import android.os.Environment
import android.widget.Toast
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale
import com.example.museumemotionapp.models.panasQuestions
import java.io.File

@Composable
fun PanasScreen(username: String, navController: NavController) {
    val context = LocalContext.current
    val scale = LocalFontScale.current.scale
    val answers = remember { mutableStateListOf<Int?>().apply { repeat(panasQuestions.size) { add(null) } } }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            Text(
                text = "Ερωτηματολόγιο 3: PANAS – Κλίμακα θετικών και αρνητικών επιδράσεων",
                style = MaterialTheme.typography.titleLarge
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Καταγράψτε το πώς νιώσατε την περασμένη εβδομάδα…",
                style = MaterialTheme.typography.bodyLarge
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Παρακαλώ, επιλέξτε έναν αριθμό από 1 (Πολύ ελαφρά ή καθόλου) έως 5 (Επαρκώς)",
                style = MaterialTheme.typography.bodyMedium
            )

            Spacer(modifier = Modifier.height(8.dp))

            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                itemsIndexed(panasQuestions) { index, question ->
                    PanasQuestionItem(
                        index = index,
                        question = question,
                        selectedAnswer = answers[index],
                        onAnswerSelected = { answers[index] = it }
                    )
                    Divider()
                }

                item {
                    Spacer(modifier = Modifier.height(16.dp))

                    Button(
                        onClick = {
                            savePanasAnswersToTxt(context, username, answers)
                            navController.navigate("artworkSelection/$username")
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Αποθήκευση Απαντήσεων")
                    }

                    Spacer(modifier = Modifier.height(16.dp))
                }
            }

            Text(
                text = "© 2025 MMAI Team | University of the Aegean",
                color = Color.Gray,
                textAlign = TextAlign.Center,
                fontSize = 12.sp * scale,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)
            )
        }
    }
}

@Composable
fun PanasQuestionItem(
    index: Int,
    question: String,
    selectedAnswer: Int?,
    onAnswerSelected: (Int) -> Unit
) {
    Column(modifier = Modifier.padding(vertical = 8.dp)) {
        Text("${index + 1}. $question", style = MaterialTheme.typography.bodyLarge)

        Row(modifier = Modifier.padding(top = 4.dp)) {
            for (i in 1..5) {
                Row(modifier = Modifier.padding(end = 8.dp)) {
                    RadioButton(
                        selected = selectedAnswer == i,
                        onClick = { onAnswerSelected(i) }
                    )
                    Text(text = "$i")
                }
            }
        }
    }
}

fun savePanasAnswersToTxt(context: Context, username: String, answers: List<Int?>) {
    try {
        val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "MuseumEmotion/$username")
        if (!dir.exists()) dir.mkdirs()

        val file = File(dir, "panas.txt")
        val content = buildString {
            panasQuestions.forEachIndexed { index, question ->
                val ans = answers[index]?.toString() ?: "Χωρίς απάντηση"
                append("${index + 1}. $question -> $ans\n")
            }
        }

        file.writeText(content)
        Toast.makeText(context, "Απαντήσεις αποθηκεύτηκαν στο ${file.absolutePath}", Toast.LENGTH_LONG).show()
    } catch (e: Exception) {
        Toast.makeText(context, "Σφάλμα αποθήκευσης: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
    }
}
