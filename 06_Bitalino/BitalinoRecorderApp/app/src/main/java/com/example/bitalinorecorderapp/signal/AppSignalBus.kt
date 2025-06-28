package com.example.bitalinorecorderapp.signal

import com.example.bitalinorecorderapp.viewmodel.SignalViewModel

object AppSignalBus {

    private var viewModel: SignalViewModel? = null

    fun registerViewModel(vm: SignalViewModel) {
        viewModel = vm
    }

    fun unregisterViewModel() {
        viewModel = null
    }

    suspend fun emit(values: List<Int>) {
        viewModel?.emitSignal(values)
    }
}
