import { useState, useEffect } from 'react';
import { Upload, BookOpen, Award, PieChart, LogOut, X, Trash2, Send } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useTranslation } from 'react-i18next';
import { useTranslationContext } from '../contexts/TranslationContext';
import DocumentViewer from './DocumentViewer';
import { LanguageSelector } from './LanguageSelector';

// Pagination Component
const Pagination = ({ totalItems, itemsPerPage, currentPage, onPageChange }) => {
  const totalPages = Math.ceil(totalItems / itemsPerPage);
  
  if (totalPages <= 1) return null;

  return (
    <div className="flex items-center justify-center space-x-2 mt-6">
      <button
        onClick={() => onPageChange(Math.max(1, currentPage - 1))}
        disabled={currentPage === 1}
        className="px-3 py-1 rounded border disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
      >
        Previous
      </button>
      
      {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
        <button
          key={page}
          onClick={() => onPageChange(page)}
          className={`px-3 py-1 rounded border ${
            currentPage === page 
              ? 'bg-blue-600 text-white' 
              : 'hover:bg-gray-50'
          }`}
        >
          {page}
        </button>
      ))}
      
      <button
        onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
        disabled={currentPage === totalPages}
        className="px-3 py-1 rounded border disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
      >
        Next
      </button>
    </div>
  );
};

// Date formatting utility
const formatDateTime = (dateString, t) => {
  const date = new Date(dateString);
  const now = new Date();
  const diffInSeconds = Math.floor((now - date) / 1000);
  
  if (diffInSeconds < 60) return t('dates.justNow');
  
  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) return t('dates.minutesAgo', { minutes: diffInMinutes });
  
  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) return t('dates.hoursAgo', { hours: diffInHours });
  
  const diffInDays = Math.floor(diffInHours / 24);
  if (diffInDays === 0) return t('dates.today');
  if (diffInDays === 1) return t('dates.yesterday');
  if (diffInDays < 30) return t('dates.daysAgo', { days: diffInDays });
  
  return date.toLocaleString();
};

// Main Component
export default function StudyAI() {
  // Translation hooks
  const { t } = useTranslation();
  const { translateDynamicContent } = useTranslationContext();
  
  // State variables
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoginView, setIsLoginView] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Navigation and data states
  const [activeTab, setActiveTab] = useState('dashboard');
  const [documents, setDocuments] = useState([]);
  const [quizzes, setQuizzes] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [showQuizModal, setShowQuizModal] = useState(false);
  const [activeQuiz, setActiveQuiz] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});
  const [quizResult, setQuizResult] = useState(null);

  // Pagination states
  const [currentDocPage, setCurrentDocPage] = useState(1);
  const [currentQuizPage, setCurrentQuizPage] = useState(1);
  const itemsPerPage = 6;

  // Analytics state
  const [analytics, setAnalytics] = useState({
    quizScores: [],
    documentProgress: [],
    recentActivity: []
  });

  const BASE_URL = 'http://127.0.0.1:5000';

  // Pagination utility
  const paginateItems = (items, currentPage, perPage) => {
    const startIndex = (currentPage - 1) * perPage;
    return items.slice(startIndex, startIndex + perPage);
  };

  // Initialize data on auth
  useEffect(() => {
    const userId = localStorage.getItem('user_id');
    if (userId) {
      setIsAuthenticated(true);
      Promise.all([
        fetchDocuments(),
        fetchQuizzes(),
        fetchAnalytics()
      ]);
    }
  }, []);

  // Auth handlers
  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch(`${BASE_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('user_id', data.user_id);
        setIsAuthenticated(true);
        await Promise.all([fetchDocuments(), fetchQuizzes(), fetchAnalytics()]);
      } else {
        setError(t('auth.loginError'));
      }
    } catch (error) {
      setError(t('auth.networkError'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      setError(t('auth.passwordMismatch'));
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await fetch(`${BASE_URL}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      const data = await response.json();

      if (response.ok) {
        await handleLogin(e);
      } else {
        setError(t('auth.registerError'));
      }
    } catch (error) {
      setError(t('auth.networkError'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('user_id');
    setIsAuthenticated(false);
    setDocuments([]);
    setQuizzes([]);
    setAnalytics({
      quizScores: [],
      documentProgress: [],
      recentActivity: []
    });
  };

  // Document handlers
  const handleFileUpload = async (file) => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', localStorage.getItem('user_id'));

    try {
      const response = await fetch(`${BASE_URL}/upload-document`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        await Promise.all([fetchDocuments(), fetchAnalytics()]);
      } else {
        const data = await response.json();
        setError(t('documents.uploadError'));
      }
    } catch (error) {
      setError(t('errors.networkError'));
    }
  };

  const handleDeleteDocument = async (documentId) => {
    if (!window.confirm(t('documents.deleteConfirm'))) {
      return;
    }

    try {
      const response = await fetch(`${BASE_URL}/delete-document/${documentId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': localStorage.getItem('user_id')
        }
      });

      if (response.ok) {
        await Promise.all([fetchDocuments(), fetchAnalytics()]);
      } else {
        setError(t('documents.deleteError'));
      }
    } catch (error) {
      setError(t('errors.networkError'));
    }
  };

  // Quiz handlers
  const handleGenerateQuiz = async (documentId) => {
    try {
      const response = await fetch(`${BASE_URL}/generate-quiz`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': localStorage.getItem('user_id')
        },
        body: JSON.stringify({
          document_id: documentId,
          num_questions: 5,
          difficulty: 'medium'
        })
      });

      if (response.ok) {
        const data = await response.json();
        // Translate quiz content
        const translatedQuiz = await translateDynamicContent(data);
        await Promise.all([fetchQuizzes(), fetchAnalytics()]);
      } else {
        setError(t('quizzes.generationError'));
      }
    } catch (error) {
      setError(t('errors.networkError'));
    }
  };

  const handleStartQuiz = async (quizId) => {
    try {
      const response = await fetch(`${BASE_URL}/get-quiz/${quizId}`, {
        headers: {
          'Authorization': localStorage.getItem('user_id')
        }
      });

      if (response.ok) {
        const quizData = await response.json();
        // Translate quiz content
        const translatedQuiz = await translateDynamicContent(quizData);
        setActiveQuiz(translatedQuiz);
        setShowQuizModal(true);
        setCurrentQuestion(0);
        setAnswers({});
        setQuizResult(null);
      } else {
        setError(t('errors.generalError'));
      }
    } catch (error) {
      setError(t('errors.networkError'));
    }
  };

  const handleSubmitQuiz = async (quizId, answers) => {
    try {
      const response = await fetch(`${BASE_URL}/submit-quiz`, {
        method: 'POST',
        headers: {
          'Authorization': localStorage.getItem('user_id'),
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          quiz_id: quizId,
          answers: answers
        })
      });

      if (response.ok) {
        const result = await response.json();
        // Translate quiz results
        const translatedResult = await translateDynamicContent(result);
        setQuizResult(translatedResult);
        await Promise.all([fetchQuizzes(), fetchAnalytics()]);
      } else {
        setError(t('quizzes.submissionError'));
      }
    } catch (error) {
      setError(t('errors.networkError'));
    }
  };

  const handleDeleteQuiz = async (quizId) => {
    if (!window.confirm(t('quizzes.deleteConfirm'))) {
      return;
    }

    try {
      const response = await fetch(`${BASE_URL}/delete-quiz/${quizId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': localStorage.getItem('user_id')
        }
      });

      if (response.ok) {
        await Promise.all([fetchQuizzes(), fetchAnalytics()]);
      } else {
        setError(t('quizzes.deleteError'));
      }
    } catch (error) {
      setError(t('errors.networkError'));
    }
  };

  // Data fetching functions
  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${BASE_URL}/get-documents`, {
        headers: {
          'Authorization': localStorage.getItem('user_id')
        }
      });

      if (response.ok) {
        const data = await response.json();
        // Translate document titles and any other text content
        const translatedDocs = await translateDynamicContent(data);
        setDocuments(translatedDocs);
      }
    } catch (error) {
      setError(t('errors.networkError'));
    }
  };

  const fetchQuizzes = async () => {
    try {
      const response = await fetch(`${BASE_URL}/get-quizzes`, {
        headers: {
          'Authorization': localStorage.getItem('user_id')
        }
      });

      if (response.ok) {
        const data = await response.json();
        // Translate quiz titles and any other text content
        const translatedQuizzes = await translateDynamicContent(data);
        setQuizzes(translatedQuizzes);
      }
    } catch (error) {
      setError(t('errors.networkError'));
    }
  };

  const fetchAnalytics = async () => {
    try {
      const response = await fetch(`${BASE_URL}/get-analytics`, {
        headers: {
          'Authorization': localStorage.getItem('user_id')
        }
      });

      if (response.ok) {
        const data = await response.json();
        // Translate analytics data
        const translatedAnalytics = await translateDynamicContent(data);
        setAnalytics(translatedAnalytics);
      }
    } catch (error) {
      setError(t('errors.networkError'));
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {!isAuthenticated ? (
        // Auth Section
        <div className="flex items-center justify-center min-h-screen p-4">
          <div className="bg-white w-full max-w-md p-8 rounded-lg shadow-lg">
            <div className="flex flex-col items-center mb-8">
              <h1 className="text-3xl font-bold text-blue-600 mb-2">{t('auth.title')}</h1>
              <p className="text-gray-600">{t('auth.subtitle')}</p>
              
              {/* Language Selector */}
              <div className="mt-4">
                <LanguageSelector />
              </div>
            </div>

            <div className="flex mb-6 border-b">
              <button
                onClick={() => setIsLoginView(true)}
                className={`flex-1 pb-4 font-medium text-sm transition-colors ${
                  isLoginView
                    ? 'border-b-2 border-blue-600 text-blue-600'
                    : 'text-gray-500'
                }`}
              >
                {t('auth.login')}
              </button>
              <button
                onClick={() => setIsLoginView(false)}
                className={`flex-1 pb-4 font-medium text-sm transition-colors ${
                  !isLoginView
                    ? 'border-b-2 border-blue-600 text-blue-600'
                    : 'text-gray-500'
                }`}
              >
                {t('auth.register')}
              </button>
            </div>

            {error && (
              <div className="bg-red-50 text-red-500 p-3 rounded mb-4 text-sm">
                {error}
              </div>
            )}

            <form onSubmit={isLoginView ? handleLogin : handleRegister} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('auth.username')}
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t('auth.password')}
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              {!isLoginView && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {t('auth.confirmPassword')}
                  </label>
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    required
                  />
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                {isLoading 
                  ? t('common.loading')
                  : isLoginView 
                    ? t('auth.loginButton')
                    : t('auth.registerButton')
                }
              </button>
            </form>
          </div>
        </div>
      ) : (
        // Main Application
        <div className="flex">
          {/* Sidebar */}
          <div className="w-64 bg-white h-screen shadow-md fixed">
            <div className="p-4">
              <h1 className="text-2xl font-bold text-blue-600">{t('auth.title')}</h1>
              <div className="mt-4">
                <LanguageSelector />
              </div>
            </div>
            <nav className="mt-8">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`flex items-center w-full px-4 py-3 ${
                  activeTab === 'dashboard' ? 'bg-blue-50 text-blue-600' : 'text-gray-600'
                }`}
              >
                <PieChart size={20} className="mr-3" />
                {t('sidebar.dashboard')}
              </button>
              <button
                onClick={() => setActiveTab('documents')}
                className={`flex items-center w-full px-4 py-3 ${
                  activeTab === 'documents' ? 'bg-blue-50 text-blue-600' : 'text-gray-600'
                }`}
              >
                <BookOpen size={20} className="mr-3" />
                {t('sidebar.documents')}
              </button>
              <button
                onClick={() => setActiveTab('quizzes')}
                className={`flex items-center w-full px-4 py-3 ${
                  activeTab === 'quizzes' ? 'bg-blue-50 text-blue-600' : 'text-gray-600'
                }`}
              >
                <Award size={20} className="mr-3" />
                {t('sidebar.quizzes')}
              </button>
              <button
                onClick={handleLogout}
                className="flex items-center w-full px-4 py-3 text-gray-600 mt-auto"
              >
                <LogOut size={20} className="mr-3" />
                {t('sidebar.logout')}
              </button>
            </nav>
          </div>

          {/* Main Content */}
          <div className="ml-64 flex-1 p-8">
            {error && (
              <div className="bg-red-50 text-red-500 p-3 rounded mb-4">
                {error}
              </div>
            )}

            {/* Dashboard Tab */}
            {activeTab === 'dashboard' && (
              <div>
                <h2 className="text-2xl font-bold mb-6">{t('dashboard.title')}</h2>
                
                {/* Stats Overview */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                  <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-sm font-medium text-gray-500">
                      {t('dashboard.stats.totalDocuments')}
                    </h3>
                    <p className="text-2xl font-bold text-gray-900 mt-2">
                      {documents.length}
                    </p>
                  </div>
                  <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-sm font-medium text-gray-500">
                      {t('dashboard.stats.completedQuizzes')}
                    </h3>
                    <p className="text-2xl font-bold text-gray-900 mt-2">
                      {quizzes.filter(q => q.score !== null).length}
                    </p>
                  </div>
                  <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-sm font-medium text-gray-500">
                      {t('dashboard.stats.averageScore')}
                    </h3>
                    <p className="text-2xl font-bold text-gray-900 mt-2">
                      {quizzes.length > 0
                        ? `${Math.round(
                            quizzes.reduce((acc, q) => acc + (q.score || 0), 0) / quizzes.length
                          )}%`
                        : 'N/A'}
                    </p>
                  </div>
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                  {/* Quiz Performance Chart */}
                  <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4">
                      {t('dashboard.charts.quizPerformance')}
                    </h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={analytics.quizScores}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis domain={[0, 100]} />
                          <Tooltip />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="score"
                            stroke="#4F46E5"
                            strokeWidth={2}
                            name={t('dashboard.charts.score')}
                            dot={{ r: 4 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Document Progress Chart */}
                  <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4">
                      {t('dashboard.charts.documentProgress')}
                    </h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={analytics.documentProgress}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="title" />
                          <YAxis domain={[0, 100]} />
                          <Tooltip />
                          <Legend />
                          <Bar
                            dataKey="progress"
                            fill="#10B981"
                            name={t('dashboard.charts.progress')}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
                {/* Recent Activity and Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4">
                      {t('dashboard.recentDocuments')}
                    </h3>
                    <div className="space-y-3">
                      {documents.slice(0, 5).map((doc) => (
                        <div key={doc.id} className="flex items-center justify-between">
                          <span className="truncate">{doc.title}</span>
                          <span className="text-sm text-gray-500 whitespace-nowrap">
                            {formatDateTime(doc.upload_date, t)}
                          </span>
                        </div>
                      ))}
                      {documents.length === 0 && (
                        <p className="text-gray-500">{t('dashboard.noDocuments')}</p>
                      )}
                    </div>
                  </div>

                  <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4">
                      {t('dashboard.recentQuizzes')}
                    </h3>
                    <div className="space-y-3">
                      {quizzes.slice(0, 5).map((quiz) => (
                        <div key={quiz.id} className="flex items-center justify-between">
                          <span className="truncate">{quiz.title}</span>
                          <span className="text-sm text-gray-500 whitespace-nowrap">
                            {quiz.score ? `${quiz.score}%` : t('quizzes.notAttempted')}
                          </span>
                        </div>
                      ))}
                      {quizzes.length === 0 && (
                        <p className="text-gray-500">{t('dashboard.noQuizzes')}</p>
                      )}
                    </div>
                  </div>

                  <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4">
                      {t('dashboard.activityFeed')}
                    </h3>
                    <div className="space-y-4">
                      {analytics.recentActivity.map((activity, index) => (
                        <div key={index} className="flex items-start space-x-3">
                          <div
                            className={`mt-1.5 w-2 h-2 rounded-full flex-shrink-0 ${
                              activity.type === 'quiz' ? 'bg-blue-500' : 'bg-green-500'
                            }`}
                          />
                          <div>
                            <p className="text-sm text-gray-900">{activity.description}</p>
                            <p className="text-xs text-gray-500">
                              {formatDateTime(activity.timestamp, t)}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Documents Tab */}
            {activeTab === 'documents' && (
              <div className="flex flex-col min-h-[calc(100vh-8rem)]">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-2xl font-bold">{t('documents.title')}</h2>
                  <label className="bg-blue-600 text-white px-4 py-2 rounded cursor-pointer hover:bg-blue-700 flex items-center">
                    <input
                      type="file"
                      className="hidden"
                      onChange={(e) => handleFileUpload(e.target.files[0])}
                      accept=".pdf,.txt,.doc,.docx"
                    />
                    <Upload size={20} className="mr-2" />
                    {t('documents.uploadDocument')}
                  </label>
                </div>
                
                <div className="flex-grow">
                  {documents.length === 0 ? (
                    <p className="text-gray-500">{t('documents.noDocuments')}</p>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {paginateItems(documents, currentDocPage, itemsPerPage).map((doc) => (
                        <div key={doc.id} className="bg-white p-6 rounded-lg shadow">
                          <div className="flex justify-between items-start mb-2">
                            <h3 className="text-lg font-semibold">{doc.title}</h3>
                            <button 
                              onClick={() => handleDeleteDocument(doc.id)}
                              className="text-gray-400 hover:text-red-600"
                            >
                              <X size={20} />
                            </button>
                          </div>
                          <p className="text-sm text-gray-500 mb-4">
                            {t('documents.uploadedAt')}: {formatDateTime(doc.upload_date, t)}
                          </p>
                          <div className="flex space-x-4">
                            <button
                              onClick={() => handleGenerateQuiz(doc.id)}
                              className="text-blue-600 hover:text-blue-700"
                            >
                              {t('documents.generateQuiz')}
                            </button>
                            <button
                              onClick={() => setSelectedDocument(doc.id)}
                              className="text-green-600 hover:text-green-700"
                            >
                              {t('documents.viewAndQuery')}
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {documents.length > itemsPerPage && (
                  <div className="mt-auto pt-6">
                    <Pagination
                      totalItems={documents.length}
                      itemsPerPage={itemsPerPage}
                      currentPage={currentDocPage}
                      onPageChange={setCurrentDocPage}
                    />
                  </div>
                )}
              </div>
            )}

            {/* Quizzes Tab */}
            {activeTab === 'quizzes' && (
              <div className="flex flex-col min-h-[calc(100vh-8rem)]">
                <h2 className="text-2xl font-bold mb-6">{t('quizzes.title')}</h2>
                
                <div className="flex-grow">
                  {quizzes.length === 0 ? (
                    <p className="text-gray-500">{t('quizzes.noQuizzes')}</p>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {paginateItems(quizzes, currentQuizPage, itemsPerPage).map((quiz) => (
                        <div key={quiz.id} className="bg-white p-6 rounded-lg shadow">
                          <div className="flex justify-between items-start mb-2">
                            <h3 className="text-lg font-semibold">{quiz.title}</h3>
                            <button 
                              onClick={() => handleDeleteQuiz(quiz.id)}
                              className="text-gray-400 hover:text-red-600"
                            >
                              <Trash2 size={20} />
                            </button>
                          </div>
                          <p className="text-sm text-gray-500 mb-4">
                            {t('quizzes.yourScore')}: {quiz.score ? `${quiz.score}%` : t('quizzes.notAttempted')}
                          </p>
                          <button
                            onClick={() => handleStartQuiz(quiz.id)}
                            className="text-blue-600 hover:text-blue-700"
                          >
                            {quiz.score ? t('quizzes.reviewQuiz') : t('quizzes.startQuiz')}
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {quizzes.length > itemsPerPage && (
                  <div className="mt-auto pt-6">
                    <Pagination
                      totalItems={quizzes.length}
                      itemsPerPage={itemsPerPage}
                      currentPage={currentQuizPage}
                      onPageChange={setCurrentQuizPage}
                    />
                  </div>
                )}
              </div>
            )}

{/* Quiz Modal */}
{showQuizModal && activeQuiz && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-40">
                <div className="bg-white p-8 rounded-lg max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-bold">{activeQuiz.title}</h3>
                    <button 
                      onClick={() => {
                        setShowQuizModal(false);
                        setActiveQuiz(null);
                        setCurrentQuestion(0);
                        setAnswers({});
                        setQuizResult(null);
                      }}
                      className="text-gray-500 hover:text-gray-700"
                    >
                      <X size={20} />
                    </button>
                  </div>

                  {quizResult ? (
                    <div className="space-y-6">
                      <div className="text-center mb-6">
                        <h4 className="text-2xl font-bold mb-2">{t('quizzes.quizComplete')}</h4>
                        <p className="text-xl">{t('quizzes.yourScore')}: {quizResult.score}%</p>
                      </div>
                      
                      <div className="space-y-6">
                        {activeQuiz.questions.map((question, index) => (
                          <div key={index} className={`p-4 rounded ${
                            quizResult.feedback[index].correct ? 'bg-green-50' : 'bg-red-50'
                          }`}>
                            <div className="mb-3">
                              <p className="font-medium text-lg mb-2">{t('quizzes.question')} {index + 1}</p>
                              <p className="mb-2">{question.text}</p>
                            </div>

                            <div className="space-y-2 mb-3">
                              {question.options.map((option, optIndex) => (
                                <div
                                  key={optIndex}
                                  className={`p-2 rounded ${
                                    answers[index] === option
                                      ? answers[index] === question.correct_answer
                                        ? 'bg-green-100'
                                        : 'bg-red-100'
                                      : option === question.correct_answer
                                        ? 'bg-green-100'
                                        : 'bg-gray-50'
                                  }`}
                                >
                                  <p className="flex items-center">
                                    {option}
                                    {option === question.correct_answer && (
                                      <span className="ml-2 text-green-600 text-sm">
                                        ({t('quizzes.correctAnswer')})
                                      </span>
                                    )}
                                    {answers[index] === option && option !== question.correct_answer && (
                                      <span className="ml-2 text-red-600 text-sm">
                                        ({t('quizzes.yourAnswer')})
                                      </span>
                                    )}
                                  </p>
                                </div>
                              ))}
                            </div>

                            <div className="mt-2 text-sm text-gray-600">
                              <p className="font-medium">{t('quizzes.explanation')}:</p>
                              <p>{quizResult.feedback[index].explanation}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                      
                      <button
                        onClick={() => {
                          setShowQuizModal(false);
                          setActiveQuiz(null);
                          setQuizResult(null);
                        }}
                        className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                      >
                        {t('common.close')}
                      </button>
                    </div>
                  ) : (
                    <>
                      <div className="mb-6">
                        <p className="text-sm text-gray-500 mb-2">
                          {t('quizzes.questionCount', {
                            current: currentQuestion + 1,
                            total: activeQuiz.questions.length
                          })}
                        </p>
                        <h4 className="text-lg font-medium mb-4">
                          {activeQuiz.questions[currentQuestion].text}
                        </h4>
                        <div className="space-y-3">
                          {activeQuiz.questions[currentQuestion].options.map((option, index) => (
                            <button
                              key={index}
                              onClick={() => {
                                setAnswers({
                                  ...answers,
                                  [currentQuestion]: option
                                });
                              }}
                              className={`w-full text-left p-3 rounded ${
                                answers[currentQuestion] === option
                                  ? 'bg-blue-50 border-blue-500'
                                  : 'border border-gray-200 hover:bg-gray-50'
                              }`}
                            >
                              {option}
                            </button>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex justify-between">
                        <button
                          onClick={() => setCurrentQuestion(prev => Math.max(0, prev - 1))}
                          disabled={currentQuestion === 0}
                          className="px-4 py-2 text-blue-600 disabled:text-gray-400"
                        >
                          {t('common.previous')}
                        </button>
                        {currentQuestion < activeQuiz.questions.length - 1 ? (
                          <button
                            onClick={() => setCurrentQuestion(prev => prev + 1)}
                            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                          >
                            {t('common.next')}
                          </button>
                        ) : (
                          <button
                            onClick={() => handleSubmitQuiz(activeQuiz.id, answers)}
                            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                          >
                            {t('quizzes.submitQuiz')}
                          </button>
                        )}
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Document Viewer Modal */}
            {selectedDocument && (
              <div className="fixed inset-0 bg-black bg-opacity-50 z-50">
                <div className="absolute inset-4 bg-white rounded-lg shadow-lg flex flex-col">
                  <div className="flex justify-between items-center p-4 border-b">
                    <h2 className="text-xl font-bold">{t('documents.queryDocument')}</h2>
                    <button 
                      onClick={() => setSelectedDocument(null)}
                      className="text-gray-500 hover:text-gray-700"
                    >
                      <X size={20} />
                    </button>
                  </div>
                  <div className="flex-1 overflow-hidden">
                    <DocumentViewer 
                      documentId={selectedDocument}
                      t={t}
                      formatDateTime={formatDateTime}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}