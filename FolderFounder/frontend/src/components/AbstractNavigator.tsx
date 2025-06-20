import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, Citation } from '../services/api';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  XCircle,
  AlertCircle,
  BarChart3,
  FileText,
  Users,
  Brain,
  // ...existing code...
  Eye,
  Calendar,
  BookOpen,
  Filter,
  // ...existing code...
} from 'lucide-react';

// Citation interface imported from api.ts

interface ScreeningResult {
  ai1_result: {
    decision: string;
    confidence: number;
    reasoning: string;
    pico_matches: Record<string, boolean>;
    quality_score: number;
    evidence_quotes: string[];
    model: string;
    strategy: string;
  };
  ai2_result: {
    decision: string;
    confidence: number;
    reasoning: string;
    pico_matches: Record<string, boolean>;
    quality_score: number;
    evidence_quotes: string[];
    model: string;
    strategy: string;
  };
  consensus: string;
  final_decision: string;
  confidence_score: number;
  human_decision?: string;
  human_notes?: string;
}

interface AbstractWithScreening {
  citation: Citation;
  screening_result?: ScreeningResult;
}

// ...existing code...

import { useParams } from 'react-router-dom';

const AbstractNavigator: React.FC = () => {
  // Get projectId from route params
  const { projectId } = useParams<{ projectId: string }>();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [filter, setFilter] = useState<'all' | 'included' | 'excluded' | 'conflicts' | 'unscreened'>('all');

  // Fetch abstracts using react-query
  const {
    data: abstracts = [],
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery<AbstractWithScreening[], any>(
    ['citations', projectId],
    () => api.getCitations(projectId),
    { retry: 1, enabled: !!projectId }
  );


  // Filter abstracts based on selected filter
  const filteredAbstracts = useMemo(() => abstracts.filter((item: AbstractWithScreening) => {
    if (filter === 'all') return true;
    if (filter === 'unscreened') return !item.screening_result;
    if (filter === 'included') return item.screening_result?.final_decision === 'include';
    if (filter === 'excluded') return item.screening_result?.final_decision === 'exclude';
    if (filter === 'conflicts') return item.screening_result?.consensus === 'dispute';
    return true;
  }), [abstracts, filter]);

  const currentAbstract = filteredAbstracts[currentIndex];


  // Navigation functions
  const goToPrevious = () => {
    setCurrentIndex((prev: number) => (prev > 0 ? prev - 1 : filteredAbstracts.length - 1));
  };

  const goToNext = () => {
    setCurrentIndex((prev: number) => (prev < filteredAbstracts.length - 1 ? prev + 1 : 0));
  };


  // Calculate metrics using useMemo
  const metrics = useMemo(() => {
    const screened = abstracts.filter((a: AbstractWithScreening) => a.screening_result).length;
    const included = abstracts.filter((a: AbstractWithScreening) => a.screening_result?.final_decision === 'include').length;
    const excluded = abstracts.filter((a: AbstractWithScreening) => a.screening_result?.final_decision === 'exclude').length;
    const uncertain = abstracts.filter((a: AbstractWithScreening) => a.screening_result?.final_decision === 'uncertain').length;
    const conflicts = abstracts.filter((a: AbstractWithScreening) => a.screening_result?.consensus === 'dispute').length;
    const humanReviewed = abstracts.filter((a: AbstractWithScreening) => a.screening_result?.human_decision).length;
    const avgConfidence = screened > 0
      ? abstracts.filter((a: AbstractWithScreening) => a.screening_result).reduce((sum: number, a: AbstractWithScreening) => sum + (a.screening_result?.confidence_score ?? 0), 0) / screened
      : 0;
    return {
      total: abstracts.length,
      screened,
      included,
      excluded,
      uncertain,
      conflicts,
      humanReviewed,
      avgConfidence,
      inclusionRate: screened > 0 ? (included / screened) * 100 : 0,
      exclusionRate: screened > 0 ? (excluded / screened) * 100 : 0,
      conflictRate: screened > 0 ? (conflicts / screened) * 100 : 0,
    };
  }, [abstracts]);


// Utility functions
  const getDecisionBadge = (decision?: string) => {
    switch (decision) {
      case 'include':
        return <Badge className="bg-green-500"><CheckCircle className="w-3 h-3 mr-1" /> Included</Badge>;
      case 'exclude':
        return <Badge className="bg-red-500"><XCircle className="w-3 h-3 mr-1" /> Excluded</Badge>;
      case 'uncertain':
        return <Badge className="bg-yellow-500"><AlertCircle className="w-3 h-3 mr-1" /> Uncertain</Badge>;
      default:
        return <Badge variant="outline">Unscreened</Badge>;
    }
  };

  // Loading and error states
  if (!projectId) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center text-red-600">
          <span className="inline-block mr-2">⚠️</span> No project selected. Please select a project to view citations.
        </div>
      </div>
    );
  }
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center text-muted-foreground">
          <span className="animate-spin inline-block mr-2">⏳</span> Loading citations...
        </div>
      </div>
    );
  }
  if (isError) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center text-red-600">
          <span className="inline-block mr-2">❌</span> Error loading citations: {error?.message ?? 'Unknown error'}
          <br />
          <button className="mt-2 underline text-blue-600" onClick={() => refetch()}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col gap-4 p-4">
      {/* Metrics Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Screening Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold">{metrics.total}</div>
              <div className="text-sm text-muted-foreground">Total Citations</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{metrics.screened}</div>
              <div className="text-sm text-muted-foreground">Screened</div>
              <Progress value={(metrics.screened / metrics.total) * 100} className="mt-1" />
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{metrics.included}</div>
              <div className="text-sm text-muted-foreground">Included</div>
              <div className="text-xs">{metrics.inclusionRate.toFixed(1)}%</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{metrics.excluded}</div>
              <div className="text-sm text-muted-foreground">Excluded</div>
              <div className="text-xs">{metrics.exclusionRate.toFixed(1)}%</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">{metrics.conflicts}</div>
              <div className="text-sm text-muted-foreground">Conflicts</div>
              <div className="text-xs">{metrics.conflictRate.toFixed(1)}%</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{metrics.avgConfidence.toFixed(0)}%</div>
              <div className="text-sm text-muted-foreground">Avg Confidence</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Navigation Controls */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={goToPrevious}
            disabled={filteredAbstracts.length === 0}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm text-muted-foreground">
            {filteredAbstracts.length > 0 ? `${currentIndex + 1} of ${filteredAbstracts.length}` : '0 of 0'}
          </span>
          <Button
            variant="outline"
            size="icon"
            onClick={goToNext}
            disabled={filteredAbstracts.length === 0}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <select
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value as typeof filter);
              setCurrentIndex(0);
            }}
            className="text-sm border rounded px-2 py-1"
            aria-label="Filter citations"
          >
            <option value="all">All Citations</option>
            <option value="unscreened">Unscreened</option>
            <option value="included">Included</option>
            <option value="excluded">Excluded</option>
            <option value="conflicts">Conflicts</option>
          </select>
        </div>
      </div>

      {/* Main Content */}
      {currentAbstract ? (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Citation Details */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <CardTitle className="text-lg">{currentAbstract.citation.title}</CardTitle>
                  <CardDescription className="mt-1">
                    <div className="flex items-center gap-2 text-sm">
                      <Users className="w-3 h-3" />
                      {currentAbstract.citation.authors}
                    </div>
                    <div className="flex items-center gap-2 text-sm mt-1">
                      <BookOpen className="w-3 h-3" />
                      {currentAbstract.citation.journal}
                      <Calendar className="w-3 h-3 ml-2" />
                      {currentAbstract.citation.year}
                    </div>
                  </CardDescription>
                </div>
                {getDecisionBadge(currentAbstract.screening_result?.final_decision)}
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-2">Abstract</h4>
                    <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                      {currentAbstract.citation.abstract}
                    </p>
                  </div>
                  
                  {currentAbstract.citation.keywords && (
                    <div>
                      <h4 className="font-semibold mb-2">Keywords</h4>
                      <div className="flex flex-wrap gap-2">
                        {currentAbstract.citation.keywords.split(';').map((keyword) => {
                          const trimmed = keyword.trim();
                          return trimmed ? (
                            <Badge key={trimmed} variant="secondary" className="text-xs">
                              {trimmed}
                            </Badge>
                          ) : null;
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Screening Results */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="w-5 h-5" />
                AI Evaluation
              </CardTitle>
            </CardHeader>
            <CardContent>
              {currentAbstract.screening_result ? (
                <Tabs defaultValue="summary" className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="summary">Summary</TabsTrigger>
                    <TabsTrigger value="ai1">AI 1</TabsTrigger>
                    <TabsTrigger value="ai2">AI 2</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="summary" className="space-y-4">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-semibold">Final Decision</span>
                        {getDecisionBadge(currentAbstract.screening_result.final_decision)}
                      </div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm">Confidence</span>
                        <span className={`font-semibold ${getConfidenceColor(currentAbstract.screening_result.confidence_score)}`}>
                          {currentAbstract.screening_result.confidence_score.toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Consensus</span>
                        <Badge variant={currentAbstract.screening_result.consensus === 'dispute' ? 'destructive' : 'default'}>
                          {currentAbstract.screening_result.consensus}
                        </Badge>
                      </div>
                    </div>
                    
                    {currentAbstract.screening_result.human_decision && (
                      <div className="pt-2 border-t">
                        <div className="flex items-center gap-2 mb-2">
                          <Eye className="w-4 h-4" />
                          <span className="text-sm font-semibold">Human Review</span>
                        </div>
                        <Badge className="mb-2">
                          {currentAbstract.screening_result.human_decision}
                        </Badge>
                        {currentAbstract.screening_result.human_notes && (
                          <p className="text-sm text-muted-foreground">
                            {currentAbstract.screening_result.human_notes}
                          </p>
                        )}
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="ai1" className="space-y-3">
                    <AIResultDetails result={currentAbstract.screening_result.ai1_result} />
                  </TabsContent>
                  
                  <TabsContent value="ai2" className="space-y-3">
                    <AIResultDetails result={currentAbstract.screening_result.ai2_result} />
                  </TabsContent>
                </Tabs>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <AlertCircle className="w-12 h-12 mx-auto mb-2" />
                  <p>Not yet screened</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-muted-foreground">
            <FileText className="w-12 h-12 mx-auto mb-2" />
            <p>No citations to display</p>
          </div>
        </div>
      )}
    </div>
  );
};

// Component for displaying individual AI results

// Use the getConfidenceColor from above

const AIResultDetails: React.FC<{ result: any }> = ({ result }) => {
  let badgeClass = 'bg-yellow-500';
  if (result.decision === 'include') badgeClass = 'bg-green-500';
  else if (result.decision === 'exclude') badgeClass = 'bg-red-500';

  return (
    <>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Model</span>
          <Badge variant="outline" className="text-xs">
            {result.model} ({result.strategy})
          </Badge>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold">Decision</span>
          <Badge className={badgeClass}>
            {result.decision}
          </Badge>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm">Confidence</span>
          <span className={`font-semibold ${getConfidenceColor(result.confidence)}`}>
            {result.confidence.toFixed(0)}%
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm">Quality Score</span>
          <span className="font-semibold">{result.quality_score.toFixed(0)}%</span>
        </div>
      </div>

      <Separator />

      <div>
        <h5 className="text-sm font-semibold mb-1">PICO Matches</h5>
        <div className="grid grid-cols-2 gap-1 text-xs">
          {Object.entries(result.pico_matches).map(([key, value]) => (
            <div key={key} className="flex items-center gap-1">
              {Boolean(value) ? (
                <CheckCircle className="w-3 h-3 text-green-500" />
              ) : (
                <XCircle className="w-3 h-3 text-red-500" />
              )}
              <span className="capitalize">{key.replace('_', ' ')}</span>
            </div>
          ))}
        </div>
      </div>

      <Separator />

      <div>
        <h5 className="text-sm font-semibold mb-1">Reasoning</h5>
        <p className="text-xs text-muted-foreground">{result.reasoning}</p>
      </div>

      {result.evidence_quotes.length > 0 && (
        <>
          <Separator />
          <div>
            <h5 className="text-sm font-semibold mb-1">Evidence Quotes</h5>
            <ul className="text-xs text-muted-foreground space-y-1">
              {result.evidence_quotes.map((quote: string) => (
                <li key={quote} className="pl-4 relative">
                  <span className="absolute left-0">"</span>
                  {quote}
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </>
  );
};

export default AbstractNavigator;
