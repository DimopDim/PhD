package com.example.museumemotionapp.screens

import android.app.Activity
import android.util.Log
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
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
        "Κατανοώ ότι μπορώ να ζητήσω να καταστραφούν οι πληροφορίες που έδωσα στο πλαίσιο της έρευνας μέχρι το χρονικό σημείο κατά το οποίο αυτά τα δεδομένα έχουν ανωνυμοποιηθεί ή έχουν ενσωματωθεί ή χρησιμοποιηθεί σε ανάλυση ή/και δημοσιευμένο υλικό.",
        "Γνωρίζω με ποιόν μπορώ να επικοινωνήσω εάν επιθυμώ περισσότερες πληροφορίες για την έρευνα",
        "Γνωρίζω σε ποιόν μπορώ να απευθυνθώ για παράπονα ή καταγγελίες",
        "Καταλαβαίνω ότι η συμμετοχή μου περιλαμβάνει τη καταγραφή βίντεο και δεδομένων από τα οποία είναι δυνατή η αναγνώρισή μου κατά την χρήση τους για την προώθηση και παρουσίαση της έρευνας, ή κατά τη δημοσιοποίηση της συλλογής δεδομένων",
        "Θέλω η ταυτότητά μου να αποκαλυφθεί σε πιθανές δημοσιεύσεις, παρουσιάσεις ή επιστημονικές αναφορές που θα προκύψουν από τη συγκεκριμένη μελέτη",
        "Γνωρίζω σε ποιόν μπορώ να απευθυνθώ για να ασκήσω τα δικαιώματά μου",
        "Καταλαβαίνω ότι το σύνολο των δεδομένων που θα προκύψει από την παρούσα έρευνα μπορεί να είναι διαθέσιμο σε τρίτους για ερευνητικούς σκοπούς"
    )

    val answers = remember { mutableStateListOf(*Array<String?>(questions.size) { null }) }
    val participantName = remember { mutableStateOf(TextFieldValue()) }
    val researcherName = remember { mutableStateOf(TextFieldValue()) }
    val date = remember { SimpleDateFormat("dd/MM/yyyy", Locale.getDefault()).format(Date()) }

    val signaturePoints = remember { mutableStateListOf<Offset>() }
    val researcherSignaturePoints = remember { mutableStateListOf<Offset>() }

    val isFormValid = answers.all { it != null } &&
            signaturePoints.any { it != Offset.Unspecified } &&
            researcherSignaturePoints.any { it != Offset.Unspecified } &&
            participantName.value.text.isNotBlank() &&
            researcherName.value.text.isNotBlank()

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

        SignatureCanvas(signaturePoints)

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

        SignatureCanvas(researcherSignaturePoints)

        Spacer(modifier = Modifier.height(8.dp))
        Button(
            onClick = { researcherSignaturePoints.clear() },
            colors = ButtonDefaults.buttonColors(containerColor = Color.LightGray)
        ) {
            Text("Καθαρισμός Υπογραφής Ερευνητή", fontSize = 14.sp * scale, color = Color.Black)
        }

        Text("Ημερομηνία: $date", fontSize = 14.sp * scale)

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = {
                saveConsentFormAsPdf(
                    context = context,
                    activity = activity,
                    username = username,
                    answers = answers.map { it ?: "" },
                    participantName = participantName.value.text,
                    researcherName = researcherName.value.text,
                    date = date,
                    signaturePoints = convertToStrokes(signaturePoints),
                    researcherSignaturePoints = convertToStrokes(researcherSignaturePoints)
                )
                navController.navigate("demographics/$username")
            },
            enabled = isFormValid
        ) {
            Text("Συνέχεια", fontSize = 16.sp * scale)
        }
    }
}

@Composable
fun SignatureCanvas(pointsList: MutableList<Offset>) {
    Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(150.dp)
            .border(1.dp, Color.Gray)
            .pointerInput(Unit) {
                detectDragGestures(
                    onDragStart = { offset ->
                        if (offset.x in 0f..size.width.toFloat() && offset.y in 0f..size.height.toFloat()) {
                            pointsList.add(Offset.Unspecified)
                            pointsList.add(offset)
                        }
                    },
                    onDrag = { change, _ ->
                        val position = change.position
                        if (position.x in 0f..size.width.toFloat() && position.y in 0f..size.height.toFloat()) {
                            pointsList.add(position)
                        }
                    }
                )
            }
    ) {
        var previous: Offset? = null
        for (point in pointsList) {
            if (point == Offset.Unspecified) {
                previous = null
            } else {
                previous?.let { p ->
                    drawLine(
                        color = Color.Black,
                        start = p,
                        end = point,
                        strokeWidth = 2f,
                        cap = StrokeCap.Round
                    )
                }
                previous = point
            }
        }
    }
}

private fun convertToStrokes(points: List<Offset>): List<List<Offset>> {
    val strokes = mutableListOf<MutableList<Offset>>()
    var currentStroke = mutableListOf<Offset>()

    for (point in points) {
        if (point == Offset.Unspecified) {
            if (currentStroke.isNotEmpty()) {
                strokes.add(currentStroke)
                currentStroke = mutableListOf()
            }
        } else {
            currentStroke.add(point)
        }
    }

    if (currentStroke.isNotEmpty()) {
        strokes.add(currentStroke)
    }

    return strokes
}
