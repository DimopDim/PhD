package com.example.museumemotionapp.screens

import android.app.Activity
import android.os.Environment
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale
import com.example.museumemotionapp.utils.saveConsentFormAsPdf
import java.text.SimpleDateFormat
import java.util.*
import android.util.Log


@Composable
fun ConsentFormScreen(navController: NavController, username: String) {
    val scale = LocalFontScale.current.scale
    val context = LocalContext.current
    val activity = context as? Activity

    val questions = listOf(
        "Έχω διαβάσει και έχω κατανοήσει το περιεχόμενο του Εντύπου Πληροφόρησης",
        "Μου δόθηκε αρκετός χρόνος για να αποφασίσω αν θέλω να συμμετέχω σε αυτή τη συζήτηση",
        "Έχω λάβει ικανοποιητικές εξηγήσεις για τη διαχείριση των προσωπικών μου δεδομένων",
        "Καταλαβαίνω ότι η συμμετοχή μου είναι εθελοντική και μπορώ να αποχωρήσω οποιαδήποτε στιγμή χωρίς να δώσω εξηγήσεις και χωρίς καμία συνέπεια",
        "Κατανοώ ότι αν αποχωρήσω από την έρευνα τα δεδομένα μου θα καταστραφούν",
        "Κατανοώ ότι μπορώ να ζητήσω να καταστραφούν οι πληροφορίες που έδωσα στο πλαίσιο της έρευνας μέχρι [ό,τι ισχύει]",
        "Γνωρίζω με ποιόν μπορώ να επικοινωνήσω εάν επιθυμώ περισσότερες πληροφορίες για την έρευνα",
        "Γνωρίζω σε ποιόν μπορώ να απευθυνθώ για παράπονα ή καταγγελίες",
        "Καταλαβαίνω ότι η συμμετοχή μου περιλαμβάνει τη καταγραφή βίντεο και δεδομένων από τα οποία είναι δυνατή η αναγνώρισή μου κατά την χρήση τους για την προώθηση και παρουσίαση της έρευνας, ή κατά τη δημοσιοποίηση της συλλογής δεδομένων",
        "Θέλω η ταυτότητά μου να αποκαλυφθεί σε πιθανές δημοσιεύσεις, παρουσιάσεις ή επιστημονικές αναφορές που θα προκύψουν από τη συγκεκριμένη μελέτη",
        "Γνωρίζω σε ποιόν μπορώ να απευθυνθώ για να ασκήσω τα δικαιώματά μου",
        "Καταλαβαίνω ότι το σύνολο των δεδομένων που θα προκύψει από την παρούσα έρευνα μπορεί να είναι διαθέσιμο σε τρίτους για ερευνητικούς σκοπούς"
    )

    val answers = remember { mutableStateListOf(*Array(questions.size) { "ΝΑΙ" }) }

    val participantName = remember { mutableStateOf(TextFieldValue()) }
    val researcherName = remember { mutableStateOf(TextFieldValue()) }
    val date = remember { SimpleDateFormat("dd/MM/yyyy", Locale.getDefault()).format(Date()) }

    val signaturePoints = remember { mutableStateListOf<Offset>() }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Spacer(modifier = Modifier.height(16.dp))

        questions.forEachIndexed { index, question ->
            Text(text = question, fontSize = 14.sp * scale)
            Row(verticalAlignment = Alignment.CenterVertically) {
                RadioButton(selected = answers[index] == "ΝΑΙ", onClick = { answers[index] = "ΝΑΙ" })
                Text("ΝΑΙ", fontSize = 14.sp * scale)
                Spacer(modifier = Modifier.width(16.dp))
                RadioButton(selected = answers[index] == "ΟΧΙ", onClick = { answers[index] = "ΟΧΙ" })
                Text("ΟΧΙ", fontSize = 14.sp * scale)
            }
            Spacer(modifier = Modifier.height(8.dp))
        }

        Spacer(modifier = Modifier.height(16.dp))
        Text("Ονοματεπώνυμο Συμμετέχοντος", fontSize = 14.sp * scale)
        OutlinedTextField(value = participantName.value, onValueChange = { participantName.value = it })

        Spacer(modifier = Modifier.height(8.dp))
        Text("Υπογραφή Συμμετέχοντος", fontSize = 14.sp * scale)

        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(150.dp)
                .border(1.dp, Color.Gray)
                .pointerInput(Unit) {
                    detectDragGestures(
                        onDragStart = { offset ->
                            signaturePoints.add(Offset.Unspecified)
                            signaturePoints.add(offset)
                        },
                        onDrag = { change, _ ->
                            val canvasSize = this.size
                            if (
                                change.position.x in 0f..canvasSize.width.toFloat() &&
                                change.position.y in 0f..canvasSize.height.toFloat()
                            ) {
                                signaturePoints.add(change.position)
                            }
                        }
                    )
                }
        ) {
            var previous: Offset? = null
            for (point in signaturePoints) {
                if (point == Offset.Unspecified) {
                    previous = null
                } else {
                    previous?.let { p1 ->
                        drawLine(
                            color = Color.Black,
                            start = p1,
                            end = point,
                            strokeWidth = 2f,
                            cap = StrokeCap.Round
                        )
                    }
                    previous = point
                }
            }
        }
        Spacer(modifier = Modifier.height(8.dp))

        Button(
            onClick = { signaturePoints.clear() },
            colors = ButtonDefaults.buttonColors(containerColor = Color.LightGray)
        ) {
            Text("Καθαρισμός Υπογραφής", fontSize = 14.sp * scale, color = Color.Black)
        }

        Spacer(modifier = Modifier.height(8.dp))

        Text("Ημερομηνία: $date", fontSize = 14.sp * scale)

        Spacer(modifier = Modifier.height(16.dp))
        Text("Ονοματεπώνυμο Ερευνητή", fontSize = 14.sp * scale)
        OutlinedTextField(value = researcherName.value, onValueChange = { researcherName.value = it })

        Spacer(modifier = Modifier.height(8.dp))
        Text("Υπογραφή Ερευνητή (χειρόγραφη)", fontSize = 14.sp * scale)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(100.dp)
                .border(1.dp, Color.Gray),
            contentAlignment = Alignment.Center
        ) {
            Text("___________", fontSize = 20.sp * scale)
        }

        Text("Ημερομηνία: $date", fontSize = 14.sp * scale)

        Spacer(modifier = Modifier.height(24.dp))

        Button(onClick = {
            Log.d("ConsentForm", "Signature points count: ${signaturePoints.size}")
            signaturePoints.forEachIndexed { index, point ->
                Log.d("ConsentForm", "Point $index: $point")
            }

            saveConsentFormAsPdf(
                context = context,
                activity = activity,
                username = username,
                answers = answers,
                participantName = participantName.value.text,
                researcherName = researcherName.value.text,
                date = date,
                signaturePoints = signaturePoints.toList()
            )
            navController.navigate("artworkSelection/$username")
        }) {
            Text("Συνέχεια", fontSize = 16.sp * scale)
        }

    }
}
