"""
Compliance Module - Regulatory compliance frameworks.

Provides comprehensive compliance management for HIPAA, GDPR, SOX, PCI-DSS,
and other regulatory frameworks relevant to enterprise quantum computing.
"""

import time
import json
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    HIPAA = "hipaa"
    GDPR = "gdpr"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    FISMA = "fisma"
    ISO27001 = "iso27001"
    NIST = "nist"
    CCPA = "ccpa"


@dataclass
class ComplianceRequirement:
    """Represents a specific compliance requirement."""
    framework: ComplianceFramework
    requirement_id: str
    title: str
    description: str
    category: str
    mandatory: bool = True
    implementation_notes: str = ""
    verification_method: str = "audit"
    responsible_team: str = "security"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceCheck:
    """Represents a compliance check result."""
    requirement_id: str
    framework: ComplianceFramework
    status: str  # 'compliant', 'non_compliant', 'partial', 'not_applicable'
    score: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_checked: float = field(default_factory=time.time)
    next_check_due: Optional[float] = None
    responsible_person: Optional[str] = None


class BaseComplianceFramework(ABC):
    """Base class for compliance framework implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.requirements: Dict[str, ComplianceRequirement] = {}
        self.check_results: Dict[str, ComplianceCheck] = {}
        self.policies: Dict[str, Any] = {}
        
        # Initialize requirements
        self._initialize_requirements()
    
    @abstractmethod
    def _initialize_requirements(self):
        """Initialize framework-specific requirements."""
        pass
    
    @abstractmethod
    def run_compliance_assessment(self) -> Dict[str, Any]:
        """Run comprehensive compliance assessment."""
        pass
    
    def add_requirement(self, requirement: ComplianceRequirement):
        """Add compliance requirement."""
        self.requirements[requirement.requirement_id] = requirement
    
    def check_requirement(self, requirement_id: str, **kwargs) -> ComplianceCheck:
        """Check specific requirement compliance."""
        requirement = self.requirements.get(requirement_id)
        if not requirement:
            raise ValueError(f"Requirement {requirement_id} not found")
        
        # Perform check based on requirement type
        check_result = self._perform_check(requirement, **kwargs)
        
        # Store result
        self.check_results[requirement_id] = check_result
        
        return check_result
    
    def _perform_check(self, requirement: ComplianceRequirement, **kwargs) -> ComplianceCheck:
        """Perform compliance check for requirement."""
        # Default implementation - override in subclasses
        return ComplianceCheck(
            requirement_id=requirement.requirement_id,
            framework=requirement.framework,
            status='not_applicable',
            score=0.0
        )
    
    def get_compliance_score(self) -> float:
        """Calculate overall compliance score."""
        if not self.check_results:
            return 0.0
        
        total_score = sum(check.score for check in self.check_results.values())
        return total_score / len(self.check_results)
    
    def get_non_compliant_items(self) -> List[ComplianceCheck]:
        """Get list of non-compliant items."""
        return [
            check for check in self.check_results.values()
            if check.status in ['non_compliant', 'partial']
        ]


class HIPAACompliance(BaseComplianceFramework):
    """HIPAA (Health Insurance Portability and Accountability Act) compliance."""
    
    def _initialize_requirements(self):
        """Initialize HIPAA requirements."""
        # Administrative Safeguards
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.HIPAA,
            requirement_id="hipaa_164.308_a_1",
            title="Security Officer",
            description="Assign a security officer responsible for security policies",
            category="administrative_safeguards",
            mandatory=True
        ))
        
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.HIPAA,
            requirement_id="hipaa_164.308_a_3",
            title="Workforce Training",
            description="Implement workforce training program for security awareness",
            category="administrative_safeguards",
            mandatory=True
        ))
        
        # Physical Safeguards
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.HIPAA,
            requirement_id="hipaa_164.310_a_1",
            title="Facility Access Controls",
            description="Implement controls to limit physical access to systems",
            category="physical_safeguards",
            mandatory=True
        ))
        
        # Technical Safeguards
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.HIPAA,
            requirement_id="hipaa_164.312_a_1",
            title="Access Control",
            description="Implement technical access controls for systems",
            category="technical_safeguards",
            mandatory=True
        ))
        
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.HIPAA,
            requirement_id="hipaa_164.312_b",
            title="Audit Controls",
            description="Implement audit controls to record access to PHI",
            category="technical_safeguards",
            mandatory=True
        ))
        
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.HIPAA,
            requirement_id="hipaa_164.312_c_1",
            title="Integrity Controls",
            description="Implement controls to ensure PHI integrity",
            category="technical_safeguards",
            mandatory=True
        ))
        
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.HIPAA,
            requirement_id="hipaa_164.312_d",
            title="Person or Entity Authentication",
            description="Implement authentication controls",
            category="technical_safeguards",
            mandatory=True
        ))
        
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.HIPAA,
            requirement_id="hipaa_164.312_e_1",
            title="Transmission Security",
            description="Implement controls for PHI transmission security",
            category="technical_safeguards",
            mandatory=True
        ))
    
    def run_compliance_assessment(self) -> Dict[str, Any]:
        """Run HIPAA compliance assessment."""
        results = {}
        
        # Check each requirement
        for req_id, requirement in self.requirements.items():
            if requirement.category == "administrative_safeguards":
                results[req_id] = self._check_administrative_safeguard(requirement)
            elif requirement.category == "physical_safeguards":
                results[req_id] = self._check_physical_safeguard(requirement)
            elif requirement.category == "technical_safeguards":
                results[req_id] = self._check_technical_safeguard(requirement)
        
        # Calculate overall score
        overall_score = self.get_compliance_score()
        
        return {
            'framework': 'HIPAA',
            'assessment_date': datetime.now().isoformat(),
            'overall_score': overall_score,
            'compliance_status': 'compliant' if overall_score >= 0.9 else 'non_compliant',
            'results': results,
            'recommendations': self._generate_hipaa_recommendations()
        }
    
    def _check_administrative_safeguard(self, requirement: ComplianceRequirement) -> ComplianceCheck:
        """Check administrative safeguard compliance."""
        if "security officer" in requirement.title.lower():
            # Check if security officer is assigned
            has_security_officer = self.config.get('security_officer_assigned', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.HIPAA,
                status='compliant' if has_security_officer else 'non_compliant',
                score=1.0 if has_security_officer else 0.0,
                evidence=['Security officer assignment documented'] if has_security_officer else [],
                findings=[] if has_security_officer else ['No security officer assigned']
            )
        
        elif "workforce training" in requirement.title.lower():
            # Check training program
            has_training = self.config.get('workforce_training_program', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.HIPAA,
                status='compliant' if has_training else 'non_compliant',
                score=1.0 if has_training else 0.0,
                evidence=['Training program documented'] if has_training else [],
                findings=[] if has_training else ['No workforce training program']
            )
        
        # Default check
        return ComplianceCheck(
            requirement_id=requirement.requirement_id,
            framework=ComplianceFramework.HIPAA,
            status='partial',
            score=0.5
        )
    
    def _check_physical_safeguard(self, requirement: ComplianceRequirement) -> ComplianceCheck:
        """Check physical safeguard compliance."""
        # For cloud/quantum systems, physical access is handled by provider
        cloud_deployment = self.config.get('cloud_deployment', True)
        
        if cloud_deployment:
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.HIPAA,
                status='compliant',
                score=1.0,
                evidence=['Cloud provider handles physical security'],
                findings=[]
            )
        
        # On-premise physical controls check
        physical_controls = self.config.get('physical_access_controls', False)
        return ComplianceCheck(
            requirement_id=requirement.requirement_id,
            framework=ComplianceFramework.HIPAA,
            status='compliant' if physical_controls else 'non_compliant',
            score=1.0 if physical_controls else 0.0
        )
    
    def _check_technical_safeguard(self, requirement: ComplianceRequirement) -> ComplianceCheck:
        """Check technical safeguard compliance."""
        if "access control" in requirement.title.lower():
            access_controls = self.config.get('access_controls_implemented', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.HIPAA,
                status='compliant' if access_controls else 'non_compliant',
                score=1.0 if access_controls else 0.0,
                evidence=['RBAC implemented', 'MFA enabled'] if access_controls else [],
                findings=[] if access_controls else ['Access controls not implemented']
            )
        
        elif "audit controls" in requirement.title.lower():
            audit_logging = self.config.get('audit_logging_enabled', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.HIPAA,
                status='compliant' if audit_logging else 'non_compliant',
                score=1.0 if audit_logging else 0.0,
                evidence=['Comprehensive audit logging'] if audit_logging else [],
                findings=[] if audit_logging else ['Audit logging not enabled']
            )
        
        elif "integrity" in requirement.title.lower():
            integrity_controls = self.config.get('data_integrity_controls', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.HIPAA,
                status='compliant' if integrity_controls else 'non_compliant',
                score=1.0 if integrity_controls else 0.0
            )
        
        elif "authentication" in requirement.title.lower():
            authentication = self.config.get('authentication_implemented', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.HIPAA,
                status='compliant' if authentication else 'non_compliant',
                score=1.0 if authentication else 0.0
            )
        
        elif "transmission security" in requirement.title.lower():
            encryption_in_transit = self.config.get('encryption_in_transit', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.HIPAA,
                status='compliant' if encryption_in_transit else 'non_compliant',
                score=1.0 if encryption_in_transit else 0.0
            )
        
        # Default
        return ComplianceCheck(
            requirement_id=requirement.requirement_id,
            framework=ComplianceFramework.HIPAA,
            status='partial',
            score=0.5
        )
    
    def _generate_hipaa_recommendations(self) -> List[str]:
        """Generate HIPAA-specific recommendations."""
        recommendations = []
        
        for check in self.get_non_compliant_items():
            if "access control" in check.requirement_id:
                recommendations.append("Implement role-based access controls with MFA")
            elif "audit" in check.requirement_id:
                recommendations.append("Enable comprehensive audit logging for all PHI access")
            elif "encryption" in check.requirement_id:
                recommendations.append("Implement encryption for data at rest and in transit")
            elif "training" in check.requirement_id:
                recommendations.append("Establish workforce security training program")
        
        return recommendations


class GDPRCompliance(BaseComplianceFramework):
    """GDPR (General Data Protection Regulation) compliance."""
    
    def _initialize_requirements(self):
        """Initialize GDPR requirements."""
        # Lawfulness, fairness and transparency
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.GDPR,
            requirement_id="gdpr_art_5_1_a",
            title="Lawfulness, Fairness and Transparency",
            description="Process personal data lawfully, fairly and transparently",
            category="data_processing_principles",
            mandatory=True
        ))
        
        # Purpose limitation
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.GDPR,
            requirement_id="gdpr_art_5_1_b",
            title="Purpose Limitation",
            description="Collect data for specified, explicit and legitimate purposes",
            category="data_processing_principles",
            mandatory=True
        ))
        
        # Data minimisation
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.GDPR,
            requirement_id="gdpr_art_5_1_c",
            title="Data Minimisation",
            description="Process only data that is adequate and necessary",
            category="data_processing_principles",
            mandatory=True
        ))
        
        # Accuracy
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.GDPR,
            requirement_id="gdpr_art_5_1_d",
            title="Accuracy",
            description="Keep personal data accurate and up to date",
            category="data_processing_principles",
            mandatory=True
        ))
        
        # Storage limitation
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.GDPR,
            requirement_id="gdpr_art_5_1_e",
            title="Storage Limitation",
            description="Store data only as long as necessary",
            category="data_processing_principles",
            mandatory=True
        ))
        
        # Integrity and confidentiality
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.GDPR,
            requirement_id="gdpr_art_5_1_f",
            title="Integrity and Confidentiality",
            description="Ensure data security and confidentiality",
            category="data_processing_principles",
            mandatory=True
        ))
        
        # Rights of data subjects
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.GDPR,
            requirement_id="gdpr_art_15",
            title="Right of Access",
            description="Provide data subjects access to their personal data",
            category="data_subject_rights",
            mandatory=True
        ))
        
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.GDPR,
            requirement_id="gdpr_art_17",
            title="Right to Erasure",
            description="Enable data subjects to request data deletion",
            category="data_subject_rights",
            mandatory=True
        ))
        
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.GDPR,
            requirement_id="gdpr_art_20",
            title="Right to Data Portability",
            description="Enable data export in machine-readable format",
            category="data_subject_rights",
            mandatory=True
        ))
    
    def run_compliance_assessment(self) -> Dict[str, Any]:
        """Run GDPR compliance assessment."""
        results = {}
        
        for req_id, requirement in self.requirements.items():
            if requirement.category == "data_processing_principles":
                results[req_id] = self._check_processing_principle(requirement)
            elif requirement.category == "data_subject_rights":
                results[req_id] = self._check_subject_right(requirement)
        
        overall_score = self.get_compliance_score()
        
        return {
            'framework': 'GDPR',
            'assessment_date': datetime.now().isoformat(),
            'overall_score': overall_score,
            'compliance_status': 'compliant' if overall_score >= 0.85 else 'non_compliant',
            'results': results,
            'recommendations': self._generate_gdpr_recommendations()
        }
    
    def _check_processing_principle(self, requirement: ComplianceRequirement) -> ComplianceCheck:
        """Check data processing principle compliance."""
        if "lawfulness" in requirement.title.lower():
            has_legal_basis = self.config.get('legal_basis_documented', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.GDPR,
                status='compliant' if has_legal_basis else 'non_compliant',
                score=1.0 if has_legal_basis else 0.0
            )
        
        elif "purpose limitation" in requirement.title.lower():
            purpose_documented = self.config.get('processing_purposes_documented', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.GDPR,
                status='compliant' if purpose_documented else 'non_compliant',
                score=1.0 if purpose_documented else 0.0
            )
        
        elif "minimisation" in requirement.title.lower():
            data_minimization = self.config.get('data_minimization_implemented', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.GDPR,
                status='compliant' if data_minimization else 'non_compliant',
                score=1.0 if data_minimization else 0.0
            )
        
        elif "integrity and confidentiality" in requirement.title.lower():
            security_measures = self.config.get('security_measures_implemented', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.GDPR,
                status='compliant' if security_measures else 'non_compliant',
                score=1.0 if security_measures else 0.0
            )
        
        # Default
        return ComplianceCheck(
            requirement_id=requirement.requirement_id,
            framework=ComplianceFramework.GDPR,
            status='partial',
            score=0.5
        )
    
    def _check_subject_right(self, requirement: ComplianceRequirement) -> ComplianceCheck:
        """Check data subject rights compliance."""
        if "access" in requirement.title.lower():
            access_mechanism = self.config.get('data_access_mechanism', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.GDPR,
                status='compliant' if access_mechanism else 'non_compliant',
                score=1.0 if access_mechanism else 0.0
            )
        
        elif "erasure" in requirement.title.lower():
            deletion_capability = self.config.get('data_deletion_capability', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.GDPR,
                status='compliant' if deletion_capability else 'non_compliant',
                score=1.0 if deletion_capability else 0.0
            )
        
        elif "portability" in requirement.title.lower():
            export_capability = self.config.get('data_export_capability', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.GDPR,
                status='compliant' if export_capability else 'non_compliant',
                score=1.0 if export_capability else 0.0
            )
        
        # Default
        return ComplianceCheck(
            requirement_id=requirement.requirement_id,
            framework=ComplianceFramework.GDPR,
            status='partial',
            score=0.5
        )
    
    def _generate_gdpr_recommendations(self) -> List[str]:
        """Generate GDPR-specific recommendations."""
        recommendations = []
        
        for check in self.get_non_compliant_items():
            if "access" in check.requirement_id:
                recommendations.append("Implement data subject access request mechanism")
            elif "erasure" in check.requirement_id:
                recommendations.append("Implement secure data deletion capabilities")
            elif "portability" in check.requirement_id:
                recommendations.append("Implement data export functionality")
            elif "security" in check.requirement_id:
                recommendations.append("Enhance security measures for personal data protection")
        
        return recommendations


class SOXCompliance(BaseComplianceFramework):
    """SOX (Sarbanes-Oxley Act) compliance for financial controls."""
    
    def _initialize_requirements(self):
        """Initialize SOX requirements."""
        # Section 302 - Corporate responsibility
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.SOX,
            requirement_id="sox_302",
            title="Corporate Responsibility for Financial Reports",
            description="CEO/CFO certification of financial report accuracy",
            category="corporate_responsibility",
            mandatory=True
        ))
        
        # Section 404 - Management assessment
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.SOX,
            requirement_id="sox_404",
            title="Management Assessment of Internal Controls",
            description="Annual assessment of internal control effectiveness",
            category="internal_controls",
            mandatory=True
        ))
        
        # IT General Controls
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.SOX,
            requirement_id="sox_itgc_access",
            title="Access Controls",
            description="Logical access controls over financial systems",
            category="it_general_controls",
            mandatory=True
        ))
        
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.SOX,
            requirement_id="sox_itgc_change",
            title="Change Management",
            description="Controls over program changes and implementations",
            category="it_general_controls",
            mandatory=True
        ))
        
        self.add_requirement(ComplianceRequirement(
            framework=ComplianceFramework.SOX,
            requirement_id="sox_itgc_operations",
            title="Computer Operations",
            description="Controls over computer operations and job scheduling",
            category="it_general_controls",
            mandatory=True
        ))
    
    def run_compliance_assessment(self) -> Dict[str, Any]:
        """Run SOX compliance assessment."""
        results = {}
        
        for req_id, requirement in self.requirements.items():
            if requirement.category == "it_general_controls":
                results[req_id] = self._check_itgc(requirement)
            elif requirement.category == "internal_controls":
                results[req_id] = self._check_internal_control(requirement)
            else:
                results[req_id] = self._check_general_requirement(requirement)
        
        overall_score = self.get_compliance_score()
        
        return {
            'framework': 'SOX',
            'assessment_date': datetime.now().isoformat(),
            'overall_score': overall_score,
            'compliance_status': 'compliant' if overall_score >= 0.95 else 'non_compliant',
            'results': results,
            'recommendations': self._generate_sox_recommendations()
        }
    
    def _check_itgc(self, requirement: ComplianceRequirement) -> ComplianceCheck:
        """Check IT General Controls compliance."""
        if "access" in requirement.title.lower():
            access_controls = self.config.get('sox_access_controls', False)
            segregation_duties = self.config.get('segregation_of_duties', False)
            
            compliant = access_controls and segregation_duties
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.SOX,
                status='compliant' if compliant else 'non_compliant',
                score=1.0 if compliant else 0.0,
                evidence=['RBAC implemented', 'Segregation of duties'] if compliant else [],
                findings=[] if compliant else ['Missing access controls or segregation of duties']
            )
        
        elif "change" in requirement.title.lower():
            change_management = self.config.get('change_management_process', False)
            approval_controls = self.config.get('change_approval_controls', False)
            
            compliant = change_management and approval_controls
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.SOX,
                status='compliant' if compliant else 'non_compliant',
                score=1.0 if compliant else 0.0
            )
        
        elif "operations" in requirement.title.lower():
            operational_controls = self.config.get('operational_controls', False)
            return ComplianceCheck(
                requirement_id=requirement.requirement_id,
                framework=ComplianceFramework.SOX,
                status='compliant' if operational_controls else 'non_compliant',
                score=1.0 if operational_controls else 0.0
            )
        
        # Default
        return ComplianceCheck(
            requirement_id=requirement.requirement_id,
            framework=ComplianceFramework.SOX,
            status='partial',
            score=0.5
        )
    
    def _check_internal_control(self, requirement: ComplianceRequirement) -> ComplianceCheck:
        """Check internal controls compliance."""
        controls_assessment = self.config.get('internal_controls_assessment', False)
        documentation = self.config.get('controls_documentation', False)
        testing = self.config.get('controls_testing', False)
        
        compliant = controls_assessment and documentation and testing
        
        return ComplianceCheck(
            requirement_id=requirement.requirement_id,
            framework=ComplianceFramework.SOX,
            status='compliant' if compliant else 'non_compliant',
            score=1.0 if compliant else 0.0,
            evidence=['Controls assessment completed', 'Documentation current', 'Testing performed'] if compliant else [],
            findings=[] if compliant else ['Missing assessment, documentation, or testing']
        )
    
    def _check_general_requirement(self, requirement: ComplianceRequirement) -> ComplianceCheck:
        """Check general SOX requirement."""
        # For corporate responsibility, assume compliance if controls are in place
        has_controls = len([r for r in self.check_results.values() if r.status == 'compliant']) > 0
        
        return ComplianceCheck(
            requirement_id=requirement.requirement_id,
            framework=ComplianceFramework.SOX,
            status='compliant' if has_controls else 'partial',
            score=1.0 if has_controls else 0.5
        )
    
    def _generate_sox_recommendations(self) -> List[str]:
        """Generate SOX-specific recommendations."""
        recommendations = []
        
        for check in self.get_non_compliant_items():
            if "access" in check.requirement_id:
                recommendations.append("Implement comprehensive access controls with segregation of duties")
            elif "change" in check.requirement_id:
                recommendations.append("Establish formal change management process with approval controls")
            elif "operations" in check.requirement_id:
                recommendations.append("Implement operational controls for system monitoring and job scheduling")
            elif "assessment" in check.requirement_id:
                recommendations.append("Conduct annual internal controls assessment and testing")
        
        return recommendations


class ComplianceManager:
    """Centralized compliance management across multiple frameworks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.frameworks: Dict[ComplianceFramework, BaseComplianceFramework] = {}
        self.enabled_frameworks = config.get('enabled_frameworks', [])
        
        # Initialize enabled frameworks
        self._initialize_frameworks()
    
    def _initialize_frameworks(self):
        """Initialize enabled compliance frameworks."""
        for framework_name in self.enabled_frameworks:
            try:
                framework = ComplianceFramework(framework_name.lower())
                
                if framework == ComplianceFramework.HIPAA:
                    self.frameworks[framework] = HIPAACompliance(self.config)
                elif framework == ComplianceFramework.GDPR:
                    self.frameworks[framework] = GDPRCompliance(self.config)
                elif framework == ComplianceFramework.SOX:
                    self.frameworks[framework] = SOXCompliance(self.config)
                # Add more frameworks as needed
                
            except ValueError:
                print(f"Unknown compliance framework: {framework_name}")
    
    def run_comprehensive_assessment(self) -> Dict[str, Any]:
        """Run assessment across all enabled frameworks."""
        results = {}
        overall_scores = {}
        
        for framework, implementation in self.frameworks.items():
            assessment = implementation.run_compliance_assessment()
            results[framework.value] = assessment
            overall_scores[framework.value] = assessment['overall_score']
        
        # Calculate weighted overall score
        if overall_scores:
            combined_score = sum(overall_scores.values()) / len(overall_scores)
        else:
            combined_score = 0.0
        
        return {
            'assessment_date': datetime.now().isoformat(),
            'enabled_frameworks': self.enabled_frameworks,
            'combined_compliance_score': combined_score,
            'framework_results': results,
            'critical_issues': self._identify_critical_issues(results),
            'priority_recommendations': self._prioritize_recommendations(results)
        }
    
    def _identify_critical_issues(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical compliance issues across frameworks."""
        critical_issues = []
        
        for framework_name, result in results.items():
            if result['compliance_status'] == 'non_compliant':
                critical_issues.append({
                    'framework': framework_name,
                    'score': result['overall_score'],
                    'severity': 'high' if result['overall_score'] < 0.5 else 'medium'
                })
        
        return critical_issues
    
    def _prioritize_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Prioritize recommendations across frameworks."""
        all_recommendations = []
        
        for framework_name, result in results.items():
            for rec in result.get('recommendations', []):
                all_recommendations.append(f"[{framework_name.upper()}] {rec}")
        
        # Simple prioritization - in practice, use more sophisticated logic
        return all_recommendations[:10]  # Top 10 recommendations
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data."""
        assessment = self.run_comprehensive_assessment()
        
        # Framework scores
        framework_scores = {}
        for framework_name, result in assessment['framework_results'].items():
            framework_scores[framework_name] = {
                'score': result['overall_score'],
                'status': result['compliance_status'],
                'last_assessment': result['assessment_date']
            }
        
        return {
            'overall_status': 'compliant' if assessment['combined_compliance_score'] >= 0.85 else 'non_compliant',
            'combined_score': assessment['combined_compliance_score'],
            'framework_scores': framework_scores,
            'critical_issues_count': len(assessment['critical_issues']),
            'recommendations_count': len(assessment['priority_recommendations']),
            'last_updated': datetime.now().isoformat()
        }