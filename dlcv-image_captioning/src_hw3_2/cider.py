import language_evaluation

def cider_score(predicts, answers):
    evaluator = language_evaluation.CocoEvaluator()
    results = evaluator.run_evaluation(predicts, answers)
    cider = results['CIDEr']
    return cider
    
    