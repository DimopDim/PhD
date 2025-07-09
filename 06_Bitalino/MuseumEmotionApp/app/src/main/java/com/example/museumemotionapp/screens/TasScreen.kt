package com.example.museumemotionapp.screens

import android.content.Context
import android.os.Environment
import android.widget.Toast
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale
import com.example.museumemotionapp.questions.tasQuestions
import java.io.File

@Composable
fun TasScreen(username: String, navController: NavController) {
    val context = LocalContext.current
    val scale = LocalFontScale.current.scale
    val answers = remember { mutableStateListOf<Int?>().apply { repeat(tasQuestions.size) { add(null) } } }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            Text(
                text = "Ερωτηματολόγιο 2: Αλεξιθυμία με την Κλίμακα Αλεξιθυμίας του Τορόντο (TAS)",
                style = MaterialTheme.typography.titleLarge
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Απαντήστε σε κάθε στοιχείο που πιστεύετε ότι περιγράφει με ακρίβεια την κατάστασή σας.",
                style = MaterialTheme.typography.bodyLarge
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                "Παρακαλώ, επιλέξτε έναν αριθμό από 1 (Διαφωνώ έντονα) έως 5 (Συμφωνώ έντονα)",
                style = MaterialTheme.typography.bodyMedium
            )

            Spacer(modifier = Modifier.height(8.dp))

            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                itemsIndexed(tasQuestions) { index, question ->
                    QuestionItem(
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
                            saveTasAnswersToTxt(context, username, answers)
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

fun saveTasAnswersToTxt(context: Context, username: String, answers: List<Int?>) {
    try {
        val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "MuseumEmotion/$username")
        if (!dir.exists()) dir.mkdirs()

        val file = File(dir, "tas.txt")
        val content = buildString {
            tasQuestions.forEachIndexed { index, question ->
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
