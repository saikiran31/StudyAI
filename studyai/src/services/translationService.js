const API_KEY = 'AIzaSyDLgCm4g-YGxaO5CKMkRQMkrycVlMjUsmE';

/**
 * Translates a single text or an array of texts into the target language.
 * @param {string|string[]} text - Text(s) to be translated.
 * @param {string} targetLanguage - Language code to translate the text into.
 * @returns {Promise<string|string[]>} Translated text(s).
 */
export const translateText = async (text, targetLanguage) => {
  try {
    // Ensure text is always sent as an array to the API
    const payload = Array.isArray(text) ? text : [text];

    const response = await fetch(
      `https://translation.googleapis.com/language/translate/v2?key=${API_KEY}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          q: payload,
          target: targetLanguage,
        }),
      }
    );

    const data = await response.json();

    if (!response.ok || !data.data) {
      throw new Error(data.error?.message || 'Translation API error');
    }

    const translations = data.data.translations.map(item => item.translatedText);

    // Return a single string if input was a string; otherwise, return array
    return Array.isArray(text) ? translations : translations[0];
  } catch (error) {
    console.error('Translation error:', error.message);
    return text; // Fallback to original input
  }
};

/**
 * Translates nested content (objects, arrays, or strings) into the target language.
 * @param {any} content - Content to translate (string, array, or object).
 * @param {string} targetLanguage - Language code to translate the content into.
 * @returns {Promise<any>} Translated content.
 */
export const translateContent = async (content, targetLanguage) => {
  if (Array.isArray(content)) {
    // Translate each item in the array
    return Promise.all(content.map(item => translateContent(item, targetLanguage)));
  }

  if (typeof content === 'object' && content !== null) {
    // Recursively translate each key-value pair in the object
    const translated = {};
    for (const key in content) {
      if (Object.prototype.hasOwnProperty.call(content, key)) {
        translated[key] = await translateContent(content[key], targetLanguage);
      }
    }
    return translated;
  }

  if (typeof content === 'string') {
    // Directly translate strings
    return translateText(content, targetLanguage);
  }

  // Return non-translatable types as-is
  return content;
};
