import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import 'fari-component-library/style'
import router from './router'
import { createPinia } from 'pinia'

const app = createApp(App)
app.use(createPinia())
app.use(router)

app.mount('#app')
