// App.jsx
import { TranslationProvider } from './contexts/TranslationContext';
import StudyAI from './components/StudyAI';
import './i18n';

function App() {
  return (
    <TranslationProvider>
      <StudyAI />
    </TranslationProvider>
  );
}

export default App