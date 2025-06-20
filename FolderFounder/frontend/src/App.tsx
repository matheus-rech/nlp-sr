import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import LandingPage from '@/components/LandingPage'
import AbstractNavigator from '@/components/AbstractNavigator'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-background scrollbar-thin">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/screening" element={<AbstractNavigator />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
          <Toaster 
            position="bottom-right"
            toastOptions={{
              className: 'bg-background border-border',
              duration: 4000,
            }}
          />
        </div>
      </Router>
    </QueryClientProvider>
  )
}

export default App