#!/usr/bin/env python3
"""
Test script to evaluate NDC Align (LSE Server) dataset queries.
Runs test queries against the API and evaluates responses.
"""

import json
import requests
import time
from typing import Dict, List, Any
from dataclasses import dataclass, field

API_URL = "http://localhost:8098/query/stream"

@dataclass
class TestQuestion:
    """A test question with expected answer components."""
    dataset: str
    query: str
    expected_components: List[str]
    question_id: str

@dataclass
class TestResult:
    """Result of a test query."""
    question_id: str
    dataset: str
    query: str
    response: str = ""
    found_components: List[str] = field(default_factory=list)
    missing_components: List[str] = field(default_factory=list)
    used_lse_data: bool = False
    score: str = "UNKNOWN"  # PASS, PARTIAL, FAIL, NO_DATA
    notes: str = ""


# Define test questions (subset of the comprehensive test suite)
TEST_QUESTIONS = [
    # Dataset 1: NDC Overview
    TestQuestion(
        dataset="NDC Overview & Domestic Comparison",
        question_id="1.1",
        query="What is Brazil's long-term climate neutrality target according to its NDC?",
        expected_components=[
            "climate neutrality by 2050",
            "all greenhouse gases",
            "Resolution 3",
            "CIM",
            "Interministerial Committee"
        ]
    ),
    TestQuestion(
        dataset="NDC Overview & Domestic Comparison",
        question_id="1.2",
        query="What are Brazil's interim emissions reduction targets for 2035?",
        expected_components=[
            "59-67%",
            "2005 levels",
            "2035",
            "CO2",
            "CH4"
        ]
    ),
    TestQuestion(
        dataset="NDC Overview & Domestic Comparison",
        question_id="1.3",
        query="Does Brazil have a formal emissions reduction target for 2030?",
        expected_components=[
            "no formal",
            "2030",
            "under development",
            "Plano Clima",
            "Resolution CIM"
        ]
    ),

    # Dataset 2: Institutions - Coordination
    TestQuestion(
        dataset="Institutions - Coordination",
        question_id="2.1",
        query="What is Brazil's main institutional body for coordinating climate policy?",
        expected_components=[
            "Interministerial Committee",
            "CIM",
            "climate change",
            "coordinate"
        ]
    ),

    # Dataset 3: Plans & Policies
    TestQuestion(
        dataset="Plans & Policies - Cross-Cutting",
        question_id="3.1",
        query="What is Brazil's main cross-cutting climate policy framework?",
        expected_components=[
            "National Policy on Climate Change",
            "PNMC",
            "Law 12.187",
            "2009"
        ]
    ),
    TestQuestion(
        dataset="Plans & Policies - Cross-Cutting",
        question_id="3.2",
        query="What is the Plano Clima and what does it include?",
        expected_components=[
            "Plano Clima",
            "National Plan",
            "Mitigation",
            "Adaptation",
            "2030"
        ]
    ),

    # Dataset 4: Sectoral Mitigation
    TestQuestion(
        dataset="Plans & Policies - Sectoral Mitigation",
        question_id="5.1",
        query="What are Brazil's policies for reducing emissions from deforestation?",
        expected_components=[
            "deforestation",
            "forest",
            "Amazon",
            "emission"
        ]
    ),

    # Dataset 5: Subnational - São Paulo
    TestQuestion(
        dataset="Subnational - São Paulo",
        question_id="6.1",
        query="Does São Paulo state have its own climate change law?",
        expected_components=[
            "São Paulo",
            "State Policy",
            "PEMC",
            "Law 13.798",
            "2009"
        ]
    ),
    TestQuestion(
        dataset="Subnational - São Paulo",
        question_id="6.3",
        query="How does São Paulo state coordinate its climate policies?",
        expected_components=[
            "São Paulo",
            "state",
            "coordinate",
            "climate"
        ]
    ),

    # Dataset 6: Subnational - Amazonas
    TestQuestion(
        dataset="Subnational - Amazonas",
        question_id="7.1",
        query="What climate policies does Amazonas state have?",
        expected_components=[
            "Amazonas",
            "state",
            "climate",
            "policy"
        ]
    ),

    # Dataset 7: Institutions - Knowledge & Evidence
    TestQuestion(
        dataset="Institutions - Knowledge & Evidence",
        question_id="9.3",
        query="How does Brazil monitor deforestation?",
        expected_components=[
            "PRODES",
            "INPE",
            "satellite",
            "deforestation"
        ]
    ),

    # Dataset 8: TPI Pathways
    TestQuestion(
        dataset="TPI Transition Pathways",
        question_id="12.1",
        query="What are the emissions pathway scenarios for Brazil according to TPI data?",
        expected_components=[
            "pathway",
            "emissions",
            "scenario",
            "TPI"
        ]
    ),
]


def query_api(query: str, timeout: int = 60) -> Dict[str, Any]:
    """Send a query to the API and collect the response."""
    try:
        response = requests.post(
            API_URL,
            json={"query": query},
            stream=True,
            timeout=timeout
        )

        full_response = ""
        sources_used = []

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode('utf-8')
            if not line_str.startswith('data: '):
                continue

            try:
                data = json.loads(line_str[6:])

                # Collect response chunks
                if data.get('type') == 'content':
                    full_response += data.get('content', '')

                # Collect tool usage info
                if data.get('type') == 'tool_use':
                    tool_name = data.get('name', '')
                    if 'LSE' in tool_name or 'lse' in tool_name.lower():
                        sources_used.append(tool_name)

            except json.JSONDecodeError:
                continue

        return {
            "response": full_response,
            "sources": sources_used,
            "success": True
        }

    except Exception as e:
        return {
            "response": "",
            "sources": [],
            "success": False,
            "error": str(e)
        }


def evaluate_response(test: TestQuestion, api_result: Dict[str, Any]) -> TestResult:
    """Evaluate an API response against expected components."""
    result = TestResult(
        question_id=test.question_id,
        dataset=test.dataset,
        query=test.query,
        response=api_result.get("response", "")
    )

    if not api_result.get("success"):
        result.score = "FAIL"
        result.notes = f"API Error: {api_result.get('error', 'Unknown')}"
        return result

    response_lower = result.response.lower()

    # Check if LSE data was used
    lse_sources = api_result.get("sources", [])
    result.used_lse_data = len(lse_sources) > 0

    # Check for expected components
    for component in test.expected_components:
        if component.lower() in response_lower:
            result.found_components.append(component)
        else:
            result.missing_components.append(component)

    # Score the response
    found_ratio = len(result.found_components) / len(test.expected_components)

    if not result.used_lse_data:
        result.score = "NO_DATA"
        result.notes = "LSE server not invoked"
    elif found_ratio >= 0.7:
        result.score = "PASS"
    elif found_ratio >= 0.4:
        result.score = "PARTIAL"
    else:
        result.score = "FAIL"

    return result


def run_tests() -> List[TestResult]:
    """Run all test questions."""
    results = []

    print(f"Running {len(TEST_QUESTIONS)} test queries...")
    print("=" * 80)

    for i, test in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] Testing: {test.dataset}")
        print(f"Query: {test.query}")

        # Query the API
        api_result = query_api(test.query)

        # Evaluate the response
        result = evaluate_response(test, api_result)
        results.append(result)

        # Print immediate feedback
        print(f"Score: {result.score}")
        if result.score == "NO_DATA":
            print(f"⚠️  WARNING: {result.notes}")
        elif result.score == "FAIL":
            print(f"❌ FAILED: Missing {len(result.missing_components)}/{len(test.expected_components)} components")
        elif result.score == "PARTIAL":
            print(f"⚠️  PARTIAL: Found {len(result.found_components)}/{len(test.expected_components)} components")
        else:
            print(f"✅ PASSED: Found {len(result.found_components)}/{len(test.expected_components)} components")

        # Rate limiting
        if i < len(TEST_QUESTIONS):
            time.sleep(2)

    return results


def generate_report(results: List[TestResult]) -> str:
    """Generate a comprehensive test report."""
    report = []
    report.append("=" * 80)
    report.append("NDC ALIGN (LSE SERVER) DATASET TEST RESULTS")
    report.append("=" * 80)
    report.append("")

    # Summary statistics
    total = len(results)
    passed = sum(1 for r in results if r.score == "PASS")
    partial = sum(1 for r in results if r.score == "PARTIAL")
    failed = sum(1 for r in results if r.score == "FAIL")
    no_data = sum(1 for r in results if r.score == "NO_DATA")

    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Total Tests: {total}")
    report.append(f"✅ PASSED:   {passed} ({passed/total*100:.1f}%)")
    report.append(f"⚠️  PARTIAL:  {partial} ({partial/total*100:.1f}%)")
    report.append(f"❌ FAILED:   {failed} ({failed/total*100:.1f}%)")
    report.append(f"🚫 NO DATA:  {no_data} ({no_data/total*100:.1f}%)")
    report.append("")

    # Overall assessment
    if no_data > total * 0.5:
        report.append("⚠️  CRITICAL: Over 50% of queries did not use LSE data!")
        report.append("This suggests the LSE server is not being invoked properly.")
    elif passed / total >= 0.8:
        report.append("✅ SUCCESS: Test suite passed with 80%+ success rate")
    elif (passed + partial) / total >= 0.8:
        report.append("⚠️  NEEDS IMPROVEMENT: Many partial matches, review response quality")
    else:
        report.append("❌ FAILURE: Less than 80% success rate")
    report.append("")

    # Detailed results by dataset
    report.append("DETAILED RESULTS BY DATASET")
    report.append("-" * 80)
    report.append("")

    # Group by dataset
    by_dataset = {}
    for result in results:
        if result.dataset not in by_dataset:
            by_dataset[result.dataset] = []
        by_dataset[result.dataset].append(result)

    for dataset, dataset_results in by_dataset.items():
        report.append(f"\n{dataset}")
        report.append("~" * len(dataset))

        for result in dataset_results:
            report.append(f"\nQuestion {result.question_id}: {result.score}")
            report.append(f"Query: {result.query}")
            report.append(f"Found: {len(result.found_components)}/{len(result.found_components) + len(result.missing_components)} components")

            if result.found_components:
                report.append(f"  ✓ Present: {', '.join(result.found_components)}")
            if result.missing_components:
                report.append(f"  ✗ Missing: {', '.join(result.missing_components)}")
            if result.notes:
                report.append(f"  Notes: {result.notes}")

            # Show a snippet of the response
            response_snippet = result.response[:200].replace('\n', ' ')
            if len(result.response) > 200:
                response_snippet += "..."
            report.append(f"  Response: {response_snippet}")

    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    return "\n".join(report)


if __name__ == "__main__":
    print("NDC Align Dataset Test Suite")
    print("Testing API responses for LSE server datasets")
    print("")

    # Run tests
    results = run_tests()

    # Generate report
    report = generate_report(results)

    # Print report
    print("\n")
    print(report)

    # Save report
    output_file = "test_scripts/ndc_align_test_results.txt"
    with open(output_file, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")
