import { useState, useEffect } from 'react';
import { Send } from 'lucide-react';

export default function DocumentViewer({ documentId }) {
  const [pdfUrl, setPdfUrl] = useState(null);
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const BASE_URL = 'http://127.0.0.1:5000';

  useEffect(() => {
    if (documentId) {
      setPdfUrl(`${BASE_URL}/get-document/${documentId}`);
    }
  }, [documentId]);

  const handleQuery = async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${BASE_URL}/query-document`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': localStorage.getItem('user_id')
        },
        body: JSON.stringify({
          document_id: documentId,
          query: query
        })
      });

      const data = await response.json();

      if (response.ok) {
        setResponse(data.response);
      } else {
        setError(data.error || 'Failed to get response');
      }
    } catch (error) {
      setError('Failed to get response');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen flex">
      {/* PDF Viewer */}
      <div className="w-1/2 h-full border-r">
        <embed 
          src={pdfUrl} 
          type="application/pdf" 
          className="w-full h-full"
        />
      </div>

      {/* Query Interface */}
      <div className="w-1/2 p-6 flex flex-col">
        <h2 className="text-xl font-bold mb-4">Ask Questions</h2>
        
        {error && (
          <div className="bg-red-50 text-red-500 p-3 rounded mb-4">
            {error}
          </div>
        )}

        <div className="flex gap-2 mb-6">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about the document..."
            className="flex-1 p-2 border rounded"
            onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
          />
          <button
            onClick={handleQuery}
            disabled={isLoading}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50 flex items-center"
          >
            {isLoading ? (
              'Loading...'
            ) : (
              <>
                <Send size={16} className="mr-2" />
                Ask
              </>
            )}
          </button>
        </div>

        {response && (
          <div className="bg-gray-50 p-4 rounded">
            <h3 className="font-medium mb-2">Response:</h3>
            <p className="text-gray-700 whitespace-pre-wrap">{response}</p>
          </div>
        )}
      </div>
    </div>
  );
}