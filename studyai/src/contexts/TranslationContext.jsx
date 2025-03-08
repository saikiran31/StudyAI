// contexts/TranslationContext.jsx
import { createContext, useContext, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { translateContent } from '../services/translationService';

const TranslationContext = createContext();

export const TranslationProvider = ({ children }) => {
  const { i18n } = useTranslation();
  const [translationCache, setTranslationCache] = useState({});

  const translateDynamicContent = useCallback(async (content, forceUpdate = false) => {
    const currentLang = i18n.language;
    const cacheKey = JSON.stringify({ content, lang: currentLang });
    
    if (!forceUpdate && translationCache[cacheKey]) {
      return translationCache[cacheKey];
    }
    
    const translated = await translateContent(content, currentLang);
    setTranslationCache(prev => ({ ...prev, [cacheKey]: translated }));
    return translated;
  }, [i18n.language, translationCache]);

  return (
    <TranslationContext.Provider value={{ translateDynamicContent }}>
      {children}
    </TranslationContext.Provider>
  );
};

export const useTranslationContext = () => useContext(TranslationContext);