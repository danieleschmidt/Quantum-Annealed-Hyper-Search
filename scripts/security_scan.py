#!/usr/bin/env python3
"""
Comprehensive security scanning for quantum hyperparameter search system.
"""

import ast
import os
import re
import json
import hashlib
import subprocess
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import importlib.util


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    category: str
    description: str
    file_path: str
    line_number: int = -1
    code_snippet: str = ""
    recommendation: str = ""
    cwe_id: Optional[str] = None


@dataclass
class SecurityScanResult:
    """Results of security scanning."""
    scan_timestamp: datetime = field(default_factory=datetime.now)
    total_files_scanned: int = 0
    issues_found: List[SecurityIssue] = field(default_factory=list)
    scan_duration_seconds: float = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        severity_counts = {}
        category_counts = {}
        
        for issue in self.issues_found:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        return {
            'total_issues': len(self.issues_found),
            'critical_issues': severity_counts.get('critical', 0),
            'high_issues': severity_counts.get('high', 0),
            'medium_issues': severity_counts.get('medium', 0),
            'low_issues': severity_counts.get('low', 0),
            'info_issues': severity_counts.get('info', 0),
            'categories': category_counts,
            'files_scanned': self.total_files_scanned,
            'scan_duration': self.scan_duration_seconds
        }


class QuantumSecurityScanner:
    """Comprehensive security scanner for quantum hyperparameter search system."""
    
    def __init__(self, project_root: str = "."):
        """Initialize security scanner."""
        self.project_root = Path(project_root)
        self.issues: List[SecurityIssue] = []
        
        # Security patterns to detect
        self.dangerous_patterns = {
            # Code injection risks
            r'eval\s*\(': {
                'severity': 'critical',
                'category': 'code_injection',
                'description': 'Use of eval() function - code injection risk',
                'cwe_id': 'CWE-95'
            },
            r'exec\s*\(': {
                'severity': 'critical', 
                'category': 'code_injection',
                'description': 'Use of exec() function - code injection risk',
                'cwe_id': 'CWE-95'
            },
            r'compile\s*\(': {
                'severity': 'high',
                'category': 'code_injection',
                'description': 'Use of compile() function - potential code injection',
                'cwe_id': 'CWE-95'
            },
            r'__import__\s*\(': {
                'severity': 'high',
                'category': 'code_injection',
                'description': 'Dynamic import usage - potential code injection',
                'cwe_id': 'CWE-95'
            },
            
            # Pickle/serialization risks
            r'pickle\.loads?\s*\(': {
                'severity': 'high',
                'category': 'deserialization',
                'description': 'Use of pickle - deserialization vulnerability',
                'cwe_id': 'CWE-502'
            },
            r'cPickle\.loads?\s*\(': {
                'severity': 'high',
                'category': 'deserialization', 
                'description': 'Use of cPickle - deserialization vulnerability',
                'cwe_id': 'CWE-502'
            },
            
            # File system risks
            r'open\s*\(\s*[\'"][^/].*\.\.[/\\]': {
                'severity': 'high',
                'category': 'path_traversal',
                'description': 'Potential path traversal vulnerability',
                'cwe_id': 'CWE-22'
            },
            
            # Subprocess risks
            r'subprocess\.(call|run|Popen)\s*\(.*shell\s*=\s*True': {
                'severity': 'high',
                'category': 'command_injection',
                'description': 'Subprocess with shell=True - command injection risk',
                'cwe_id': 'CWE-78'
            },
            r'os\.system\s*\(': {
                'severity': 'high',
                'category': 'command_injection',
                'description': 'Use of os.system() - command injection risk',
                'cwe_id': 'CWE-78'
            },
            
            # Cryptographic issues
            r'hashlib\.md5\s*\(': {
                'severity': 'medium',
                'category': 'weak_crypto',
                'description': 'Use of MD5 - cryptographically weak',
                'cwe_id': 'CWE-327'
            },
            r'hashlib\.sha1\s*\(': {
                'severity': 'medium',
                'category': 'weak_crypto',
                'description': 'Use of SHA1 - cryptographically weak',
                'cwe_id': 'CWE-327'
            },
            
            # Hardcoded secrets
            r'password\s*=\s*[\'"][^\'"]+[\'"]': {
                'severity': 'high',
                'category': 'hardcoded_secret',
                'description': 'Potential hardcoded password',
                'cwe_id': 'CWE-798'
            },
            r'api_key\s*=\s*[\'"][^\'"]+[\'"]': {
                'severity': 'high',
                'category': 'hardcoded_secret',
                'description': 'Potential hardcoded API key',
                'cwe_id': 'CWE-798'
            },
            r'secret\s*=\s*[\'"][^\'"]+[\'"]': {
                'severity': 'high',
                'category': 'hardcoded_secret',
                'description': 'Potential hardcoded secret',
                'cwe_id': 'CWE-798'
            },
            
            # SQL injection (less relevant for this project but good to check)
            r'\.execute\s*\(\s*[\'"][^\'\"]*%s[^\'\"]*[\'"]': {
                'severity': 'high',
                'category': 'sql_injection',
                'description': 'Potential SQL injection vulnerability',
                'cwe_id': 'CWE-89'
            }
        }
        
        # Quantum-specific security patterns
        self.quantum_patterns = {
            r'backend\s*=\s*[\'"]dwave[\'"]': {
                'severity': 'info',
                'category': 'quantum_config',
                'description': 'D-Wave backend usage - ensure proper authentication'
            },
            r'quantum_token\s*=': {
                'severity': 'medium',
                'category': 'quantum_secret',
                'description': 'Quantum service token - ensure not hardcoded'
            },
            r'solver\s*=\s*[\'"][^\'\"]*real[^\'\"]*[\'"]': {
                'severity': 'info',
                'category': 'quantum_config',
                'description': 'Real quantum hardware usage - ensure proper access controls'
            }
        }
    
    def scan_all(self) -> SecurityScanResult:
        """Run comprehensive security scan."""
        import time
        start_time = time.time()
        
        print("🔒 Starting Quantum Security Scan")
        print("=" * 40)
        
        self.issues = []
        files_scanned = 0
        
        # Scan Python files
        python_files = list(self.project_root.rglob("*.py"))
        for py_file in python_files:
            try:
                self.scan_python_file(py_file)
                files_scanned += 1
            except Exception as e:
                print(f"⚠️  Error scanning {py_file}: {e}")
        
        # Scan configuration files
        config_files = []
        for pattern in ["*.json", "*.yaml", "*.yml", "*.toml", "*.cfg", "*.ini"]:
            config_files.extend(self.project_root.rglob(pattern))
        
        for config_file in config_files:
            try:
                self.scan_config_file(config_file)
                files_scanned += 1
            except Exception as e:
                print(f"⚠️  Error scanning {config_file}: {e}")
        
        # Additional security checks
        self.check_file_permissions()
        self.check_dependencies()
        
        duration = time.time() - start_time
        
        result = SecurityScanResult(
            total_files_scanned=files_scanned,
            issues_found=self.issues,
            scan_duration_seconds=duration
        )
        
        return result
    
    def scan_python_file(self, file_path: Path) -> None:
        """Scan Python file for security issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            self.add_issue(
                severity='low',
                category='file_access',
                description=f'Could not read file: {e}',
                file_path=str(file_path)
            )
            return
        
        # Pattern-based scanning
        all_patterns = {**self.dangerous_patterns, **self.quantum_patterns}
        
        for line_num, line in enumerate(lines, 1):
            for pattern, config in all_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    self.add_issue(
                        severity=config['severity'],
                        category=config['category'],
                        description=config['description'],
                        file_path=str(file_path),
                        line_number=line_num,
                        code_snippet=line.strip(),
                        cwe_id=config.get('cwe_id')
                    )
        
        # AST-based analysis
        try:
            tree = ast.parse(content)
            self.analyze_ast(tree, file_path)
        except SyntaxError as e:
            self.add_issue(
                severity='low',
                category='syntax_error',
                description=f'Syntax error in file: {e}',
                file_path=str(file_path),
                line_number=getattr(e, 'lineno', -1)
            )
    
    def analyze_ast(self, tree: ast.AST, file_path: Path) -> None:
        """Analyze Python AST for security issues."""
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, scanner):
                self.scanner = scanner
                self.file_path = file_path
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        self.scanner.add_issue(
                            severity='critical',
                            category='dangerous_function',
                            description=f'Use of dangerous function: {node.func.id}',
                            file_path=str(self.file_path),
                            line_number=node.lineno,
                            cwe_id='CWE-95'
                        )
                
                # Check for subprocess calls
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'subprocess'):
                        # Check for shell=True
                        for keyword in node.keywords:
                            if (keyword.arg == 'shell' and 
                                isinstance(keyword.value, ast.Constant) and
                                keyword.value.value is True):
                                self.scanner.add_issue(
                                    severity='high',
                                    category='command_injection',
                                    description='subprocess call with shell=True',
                                    file_path=str(self.file_path),
                                    line_number=node.lineno,
                                    cwe_id='CWE-78'
                                )
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for dangerous imports
                for alias in node.names:
                    if alias.name in ['pickle', 'cPickle', 'marshal']:
                        self.scanner.add_issue(
                            severity='medium',
                            category='dangerous_import',
                            description=f'Import of potentially dangerous module: {alias.name}',
                            file_path=str(self.file_path),
                            line_number=node.lineno,
                            cwe_id='CWE-502'
                        )
                
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                # Check for hardcoded secrets in assignments
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    value = node.value.value
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id.lower()
                            if any(keyword in var_name for keyword in 
                                   ['password', 'secret', 'token', 'key', 'api']):
                                if len(value) > 8:  # Likely not a placeholder
                                    self.scanner.add_issue(
                                        severity='high',
                                        category='hardcoded_secret',
                                        description=f'Potential hardcoded secret in variable: {target.id}',
                                        file_path=str(self.file_path),
                                        line_number=node.lineno,
                                        cwe_id='CWE-798'
                                    )
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor(self)
        visitor.visit(tree)
    
    def scan_config_file(self, file_path: Path) -> None:
        """Scan configuration file for security issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return
        
        # Check for secrets in config files
        secret_patterns = [
            r'password\s*[:=]\s*[\'"]?[^\s\'"]+[\'"]?',
            r'api_key\s*[:=]\s*[\'"]?[^\s\'"]+[\'"]?',
            r'secret\s*[:=]\s*[\'"]?[^\s\'"]+[\'"]?',
            r'token\s*[:=]\s*[\'"]?[^\s\'"]+[\'"]?'
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.add_issue(
                        severity='high',
                        category='config_secret',
                        description='Potential secret in configuration file',
                        file_path=str(file_path),
                        line_number=line_num,
                        code_snippet=line.strip(),
                        cwe_id='CWE-798'
                    )
    
    def check_file_permissions(self) -> None:
        """Check file permissions for security issues."""
        sensitive_files = [
            '*.key', '*.pem', '*.p12', '*.pfx',
            '.env', '.secret', 'secrets.txt'
        ]
        
        for pattern in sensitive_files:
            for file_path in self.project_root.rglob(pattern):
                try:
                    stat = file_path.stat()
                    # Check if file is world-readable (Unix systems)
                    if hasattr(stat, 'st_mode') and stat.st_mode & 0o044:
                        self.add_issue(
                            severity='medium',
                            category='file_permissions',
                            description='Sensitive file is world-readable',
                            file_path=str(file_path),
                            cwe_id='CWE-732'
                        )
                except Exception:
                    continue
    
    def check_dependencies(self) -> None:
        """Check dependencies for known vulnerabilities."""
        # Check requirements.txt
        req_files = ['requirements.txt', 'requirements-dev.txt', 'pyproject.toml']
        
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r') as f:
                        content = f.read()
                    
                    # Look for unpinned dependencies
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Check for unpinned versions
                            if '==' not in line and '>=' not in line and '~=' not in line:
                                if any(char in line for char in ['<', '>', '!']):
                                    continue  # Has some version constraint
                                
                                self.add_issue(
                                    severity='low',
                                    category='dependency_management',
                                    description='Unpinned dependency version',
                                    file_path=str(req_path),
                                    line_number=line_num,
                                    code_snippet=line,
                                    cwe_id='CWE-1104'
                                )
                
                except Exception:
                    continue
    
    def add_issue(self, severity: str, category: str, description: str, 
                  file_path: str, line_number: int = -1, code_snippet: str = "",
                  recommendation: str = "", cwe_id: Optional[str] = None) -> None:
        """Add security issue to results."""
        issue = SecurityIssue(
            severity=severity,
            category=category,
            description=description,
            file_path=file_path,
            line_number=line_number,
            code_snippet=code_snippet,
            recommendation=recommendation,
            cwe_id=cwe_id
        )
        self.issues.append(issue)
    
    def generate_report(self, result: SecurityScanResult, output_file: str = "security_report.json") -> None:
        """Generate comprehensive security report."""
        summary = result.get_summary()
        
        # Create detailed report
        report = {
            'scan_metadata': {
                'timestamp': result.scan_timestamp.isoformat(),
                'duration_seconds': result.scan_duration_seconds,
                'files_scanned': result.total_files_scanned,
                'scanner_version': '1.0.0'
            },
            'summary': summary,
            'issues': []
        }
        
        # Add issues
        for issue in result.issues_found:
            issue_dict = {
                'severity': issue.severity,
                'category': issue.category,
                'description': issue.description,
                'file_path': issue.file_path,
                'line_number': issue.line_number,
                'code_snippet': issue.code_snippet,
                'recommendation': issue.recommendation,
                'cwe_id': issue.cwe_id
            }
            report['issues'].append(issue_dict)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Security report saved to: {output_file}")
    
    def print_summary(self, result: SecurityScanResult) -> None:
        """Print security scan summary."""
        summary = result.get_summary()
        
        print(f"\n🔒 Security Scan Summary")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📁 Files scanned: {summary['files_scanned']}")
        print(f"⏱️  Duration: {summary['scan_duration']:.2f}s")
        print(f"🚨 Total issues: {summary['total_issues']}")
        
        if summary['total_issues'] > 0:
            print(f"\n📊 Issues by severity:")
            print(f"  🔴 Critical: {summary['critical_issues']}")
            print(f"  🟠 High: {summary['high_issues']}")
            print(f"  🟡 Medium: {summary['medium_issues']}")
            print(f"  🔵 Low: {summary['low_issues']}")
            print(f"  ℹ️  Info: {summary['info_issues']}")
            
            print(f"\n📋 Issues by category:")
            for category, count in summary['categories'].items():
                print(f"  • {category}: {count}")
        
        # Security recommendations
        if summary['critical_issues'] > 0:
            print(f"\n⚠️  CRITICAL: Immediate action required!")
        elif summary['high_issues'] > 0:
            print(f"\n⚠️  HIGH: Review and fix high-severity issues")
        elif summary['medium_issues'] > 0:
            print(f"\n✅ Good security posture, minor improvements needed")
        else:
            print(f"\n🎉 Excellent! No critical security issues found")


def main():
    """Run security scan."""
    scanner = QuantumSecurityScanner()
    result = scanner.scan_all()
    
    # Print summary
    scanner.print_summary(result) 
    
    # Generate detailed report
    scanner.generate_report(result, "security_report.json")
    
    return result


if __name__ == "__main__":
    result = main()