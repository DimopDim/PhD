package com.example.bitalinorecorderapp.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow

class SignalViewModel : ViewModel() {

    private val _analogSignalFlow = MutableSharedFlow<List<Int>>(replay = 1)
    val analogSignalFlow: SharedFlow<List<Int>> = _analogSignalFlow

    suspend fun emitSignal(values: List<Int>) {
        _analogSignalFlow.emit(values)
    }
}
