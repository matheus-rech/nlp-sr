import React from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  IconBrain, 
  IconRocket, 
  IconShieldCheck, 
  IconSparkles,
  IconDatabase,
  IconCloudComputing,
  IconChartBar,
  IconUsers
} from '@tabler/icons-react'

const LandingPage: React.FC = () => {
  const navigate = useNavigate()

  const features = [
    {
      icon: IconBrain,
      title: "AI-Powered Screening",
      description: "Leverage multiple LLM providers for intelligent citation screening",
      gradient: "from-purple-600 to-blue-600"
    },
    {
      icon: IconShieldCheck,
      title: "Dual AI Validation",
      description: "Two-model evaluation system ensures accuracy and reduces bias",
      gradient: "from-green-600 to-teal-600"
    },
    {
      icon: IconDatabase,
      title: "Multi-Format Support",
      description: "Import citations from RIS, XML, EndNote, BibTeX, and more",
      gradient: "from-orange-600 to-red-600"
    },
    {
      icon: IconChartBar,
      title: "Real-time Analytics",
      description: "Track screening progress with beautiful visualizations",
      gradient: "from-blue-600 to-indigo-600"
    }
  ]

  const stats = [
    { value: "99.9%", label: "Uptime", icon: IconCloudComputing },
    { value: "10k+", label: "Citations Processed", icon: IconDatabase },
    { value: "8+", label: "LLM Providers", icon: IconBrain },
    { value: "500+", label: "Happy Researchers", icon: IconUsers }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-gray-100/50 dark:bg-grid-gray-800/50" />
        <motion.div
          className="absolute inset-0"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2 }}
        >
          <div className="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-purple-500/20 blur-3xl" />
          <div className="absolute -bottom-40 -left-40 h-80 w-80 rounded-full bg-blue-500/20 blur-3xl" />
        </motion.div>

        <div className="relative container mx-auto px-4 py-24">
          <motion.div
            className="text-center max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.div
              className="inline-flex items-center gap-2 mb-6"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", delay: 0.2 }}
            >
              <Badge variant="gradient" className="px-4 py-1 text-sm">
                <IconSparkles className="w-4 h-4 mr-1" />
                AI-Powered Research Tool
              </Badge>
            </motion.div>

            <motion.h1
              className="text-6xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 dark:from-gray-100 dark:to-gray-400 bg-clip-text text-transparent mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
            >
              Otto-SR: Smart Citation Screening
            </motion.h1>

            <motion.p
              className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              Accelerate your systematic reviews with AI-powered citation screening. 
              Support for 8+ LLM providers, dual-model validation, and real-time analytics.
            </motion.p>

            <motion.div
              className="flex gap-4 justify-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.5 }}
            >
              <Button 
                size="lg" 
                variant="gradient"
                onClick={() => navigate('/screening')}
                className="text-lg px-8"
              >
                <IconRocket className="mr-2 h-5 w-5" />
                Start Screening
              </Button>
              <Button 
                size="lg" 
                variant="outline"
                className="text-lg px-8"
              >
                View Demo
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 border-y border-gray-200 dark:border-gray-700">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                className="text-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <stat.icon className="w-8 h-8 mx-auto mb-2 text-primary" />
                <motion.div
                  className="text-4xl font-bold text-gray-900 dark:text-gray-100"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ 
                    type: "spring", 
                    stiffness: 200, 
                    delay: 0.8 + index * 0.1 
                  }}
                >
                  {stat.value}
                </motion.div>
                <div className="text-gray-600 dark:text-gray-400">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24">
        <div className="container mx-auto px-4">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl font-bold mb-4">Powerful Features</h2>
            <p className="text-xl text-gray-600 dark:text-gray-400">
              Everything you need for efficient systematic reviews
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card hoverable className="h-full">
                  <CardHeader>
                    <motion.div
                      className={`w-12 h-12 rounded-lg bg-gradient-to-r ${feature.gradient} flex items-center justify-center mb-4`}
                      whileHover={{ rotate: 360 }}
                      transition={{ duration: 0.5 }}
                    >
                      <feature.icon className="w-6 h-6 text-white" />
                    </motion.div>
                    <CardTitle className="text-xl">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-base">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-r from-purple-600 to-blue-600 text-white">
        <div className="container mx-auto px-4 text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl font-bold mb-4">Ready to Transform Your Research?</h2>
            <p className="text-xl mb-8 opacity-90">
              Join thousands of researchers using AI to accelerate their systematic reviews
            </p>
            <Button 
              size="lg" 
              variant="secondary"
              onClick={() => navigate('/screening')}
              className="text-lg px-8"
            >
              Get Started Free
            </Button>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default LandingPage