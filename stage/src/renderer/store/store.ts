import { configureStore } from "@reduxjs/toolkit"
import { Action } from "redux"
import { ThunkAction } from "redux-thunk"

import fileReducer from "./fileSlice"


const makeStore = () => {
    return configureStore({
        reducer: { file: fileReducer }
    })
}

const store = makeStore()

export type AppState = ReturnType<typeof store.getState>

export type AppDispatch = typeof store.dispatch

export type AppThunk<ReturnType = void> = ThunkAction<
    ReturnType,
    AppState,
    unknown,
    Action<string>
>

export default store
