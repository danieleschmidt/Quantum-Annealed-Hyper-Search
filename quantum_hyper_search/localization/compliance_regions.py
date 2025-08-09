"""
Compliance Region Manager

Provides region-specific compliance management for quantum hyperparameter optimization,
including data residency, privacy regulations, and jurisdictional requirements.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance levels for different requirements."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class RegulationRequirement:
    """Represents a specific regulatory requirement."""
    regulation_id: str
    name: str
    description: str
    compliance_level: ComplianceLevel
    data_types: List[str]
    geographic_scope: List[str]
    retention_period_days: Optional[int] = None
    encryption_required: bool = True
    audit_logging_required: bool = True
    consent_required: bool = False
    right_to_erasure: bool = False
    data_portability: bool = False
    breach_notification_hours: Optional[int] = None
    penalties: Optional[str] = None


@dataclass 
class ComplianceRegion:
    """Represents a compliance region with specific requirements."""
    region_id: str
    name: str
    country_codes: List[str]
    regulations: List[RegulationRequirement]
    data_residency_required: bool = True
    allowed_data_transfers: List[str] = field(default_factory=list)
    quantum_computing_restrictions: List[str] = field(default_factory=list)
    encryption_standards: List[str] = field(default_factory=list)
    audit_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def get_applicable_regulations(self, data_classification: DataClassification) -> List[RegulationRequirement]:
        """Get regulations applicable to specific data classification."""
        applicable = []
        for reg in self.regulations:
            if data_classification.value in reg.data_types or 'all' in reg.data_types:
                applicable.append(reg)
        return applicable
    
    def requires_local_processing(self) -> bool:
        """Check if local data processing is required."""
        return self.data_residency_required
    
    def get_max_retention_days(self, data_classification: DataClassification) -> Optional[int]:
        """Get maximum data retention period for data classification."""
        applicable_regs = self.get_applicable_regulations(data_classification)
        retention_periods = [reg.retention_period_days for reg in applicable_regs 
                           if reg.retention_period_days is not None]
        
        return min(retention_periods) if retention_periods else None


class ComplianceRegionManager:
    """
    Manages compliance requirements across different geographic regions
    for quantum hyperparameter optimization systems.
    
    Provides region-specific compliance checks, data handling requirements,
    and regulatory compliance validation.
    """
    
    def __init__(self):
        """Initialize compliance region manager."""
        self.regions: Dict[str, ComplianceRegion] = {}
        self.current_region: Optional[str] = None
        self.compliance_cache: Dict[str, Any] = {}
        
        # Initialize default compliance regions
        self._initialize_default_regions()
        
        logger.info("Compliance region manager initialized")
    
    def _initialize_default_regions(self):
        """Initialize default compliance regions and regulations."""
        
        # GDPR (European Union)
        gdpr = RegulationRequirement(
            regulation_id="gdpr",
            name="General Data Protection Regulation",
            description="EU data protection regulation",
            compliance_level=ComplianceLevel.CRITICAL,
            data_types=["personal", "sensitive", "all"],
            geographic_scope=["EU"],
            retention_period_days=2555,  # 7 years max
            encryption_required=True,
            audit_logging_required=True,
            consent_required=True,
            right_to_erasure=True,
            data_portability=True,
            breach_notification_hours=72,
            penalties="Up to 4% of annual turnover or €20 million"
        )
        
        eu_region = ComplianceRegion(
            region_id="eu",
            name="European Union",
            country_codes=[
                "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", 
                "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", 
                "PL", "PT", "RO", "SK", "SI", "ES", "SE"
            ],
            regulations=[gdpr],
            data_residency_required=True,
            allowed_data_transfers=["adequacy_decision", "binding_corporate_rules", "standard_contractual_clauses"],
            encryption_standards=["AES-256", "RSA-2048"],
            audit_requirements={
                "data_protection_officer": True,
                "privacy_impact_assessment": True,
                "regular_audits": True
            }
        )
        
        # CCPA (California)
        ccpa = RegulationRequirement(
            regulation_id="ccpa",
            name="California Consumer Privacy Act",
            description="California privacy regulation",
            compliance_level=ComplianceLevel.HIGH,
            data_types=["personal", "all"],
            geographic_scope=["US-CA"],
            retention_period_days=1095,  # 3 years typical
            encryption_required=True,
            audit_logging_required=True,
            consent_required=False,
            right_to_erasure=True,
            data_portability=True,
            breach_notification_hours=None,
            penalties="Up to $7,500 per violation"
        )
        
        california_region = ComplianceRegion(
            region_id="us_california",
            name="California, United States",
            country_codes=["US"],
            regulations=[ccpa],
            data_residency_required=False,
            allowed_data_transfers=["us_states", "adequacy_countries"],
            encryption_standards=["AES-256", "RSA-2048"],
            audit_requirements={"consumer_request_log": True}
        )
        
        # HIPAA (US Healthcare)
        hipaa = RegulationRequirement(
            regulation_id="hipaa",
            name="Health Insurance Portability and Accountability Act",
            description="US healthcare privacy regulation",
            compliance_level=ComplianceLevel.CRITICAL,
            data_types=["health", "medical", "phi"],
            geographic_scope=["US"],
            retention_period_days=2555,  # 7 years for medical records
            encryption_required=True,
            audit_logging_required=True,
            consent_required=True,
            right_to_erasure=False,
            data_portability=True,
            breach_notification_hours=72,
            penalties="Up to $1.5 million per incident"
        )
        
        us_healthcare_region = ComplianceRegion(
            region_id="us_healthcare",
            name="United States Healthcare",
            country_codes=["US"],
            regulations=[hipaa],
            data_residency_required=True,
            allowed_data_transfers=["business_associate_agreements"],
            quantum_computing_restrictions=["no_phi_on_public_quantum"],
            encryption_standards=["FIPS-140-2", "AES-256"],
            audit_requirements={
                "access_controls": True,
                "audit_logs": True,
                "incident_response": True,
                "risk_assessment": True
            }
        )
        
        # SOX (US Financial)
        sox = RegulationRequirement(
            regulation_id="sox",
            name="Sarbanes-Oxley Act",
            description="US financial reporting regulation",
            compliance_level=ComplianceLevel.HIGH,
            data_types=["financial", "corporate"],
            geographic_scope=["US"],
            retention_period_days=2555,  # 7 years
            encryption_required=True,
            audit_logging_required=True,
            consent_required=False,
            right_to_erasure=False,
            data_portability=False,
            penalties="Criminal penalties up to $25 million and 20 years imprisonment"
        )
        
        us_financial_region = ComplianceRegion(
            region_id="us_financial",
            name="United States Financial",
            country_codes=["US"],
            regulations=[sox],
            data_residency_required=True,
            allowed_data_transfers=["regulatory_approval"],
            encryption_standards=["FIPS-140-2", "AES-256"],
            audit_requirements={
                "segregation_of_duties": True,
                "change_management": True,
                "access_reviews": True,
                "immutable_audit_trail": True
            }
        )
        
        # China Cybersecurity Law
        csl = RegulationRequirement(
            regulation_id="china_csl",
            name="China Cybersecurity Law",
            description="China cybersecurity and data protection regulation",
            compliance_level=ComplianceLevel.CRITICAL,
            data_types=["personal", "important", "all"],
            geographic_scope=["CN"],
            retention_period_days=None,  # Varies by data type
            encryption_required=True,
            audit_logging_required=True,
            consent_required=True,
            right_to_erasure=False,
            data_portability=False,
            penalties="Significant fines and business suspension"
        )
        
        china_region = ComplianceRegion(
            region_id="china",
            name="China",
            country_codes=["CN"],
            regulations=[csl],
            data_residency_required=True,
            allowed_data_transfers=[],  # Very restricted
            quantum_computing_restrictions=["government_approval_required"],
            encryption_standards=["SM4", "AES-256"],
            audit_requirements={
                "government_reporting": True,
                "local_representative": True,
                "security_assessment": True
            }
        )
        
        # Japan APPI
        appi = RegulationRequirement(
            regulation_id="japan_appi",
            name="Act on Protection of Personal Information",
            description="Japan personal information protection law",
            compliance_level=ComplianceLevel.HIGH,
            data_types=["personal", "sensitive"],
            geographic_scope=["JP"],
            retention_period_days=1095,  # Typically 3 years
            encryption_required=True,
            audit_logging_required=True,
            consent_required=True,
            right_to_erasure=True,
            data_portability=False,
            penalties="Up to 1 billion yen"
        )
        
        japan_region = ComplianceRegion(
            region_id="japan",
            name="Japan",
            country_codes=["JP"],
            regulations=[appi],
            data_residency_required=False,
            allowed_data_transfers=["adequacy_countries", "consent"],
            encryption_standards=["AES-256"],
            audit_requirements={"privacy_officer": True}
        )
        
        # Add regions to manager
        self.regions = {
            "eu": eu_region,
            "us_california": california_region,
            "us_healthcare": us_healthcare_region,
            "us_financial": us_financial_region,
            "china": china_region,
            "japan": japan_region
        }
    
    def set_region(self, region_id: str) -> bool:
        """Set the current compliance region."""
        if region_id in self.regions:
            self.current_region = region_id
            logger.info(f"Compliance region set to: {region_id}")
            return True
        else:
            logger.warning(f"Unknown compliance region: {region_id}")
            return False
    
    def get_current_region(self) -> Optional[ComplianceRegion]:
        """Get the current compliance region."""
        if self.current_region:
            return self.regions.get(self.current_region)
        return None
    
    def get_available_regions(self) -> List[str]:
        """Get list of available compliance regions."""
        return list(self.regions.keys())
    
    def get_region_info(self, region_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a compliance region."""
        region = self.regions.get(region_id)
        if not region:
            return None
        
        return {
            'region_id': region.region_id,
            'name': region.name,
            'country_codes': region.country_codes,
            'regulations': [
                {
                    'id': reg.regulation_id,
                    'name': reg.name,
                    'description': reg.description,
                    'compliance_level': reg.compliance_level.value,
                    'data_types': reg.data_types,
                    'encryption_required': reg.encryption_required,
                    'consent_required': reg.consent_required,
                    'right_to_erasure': reg.right_to_erasure,
                    'data_portability': reg.data_portability,
                    'retention_days': reg.retention_period_days,
                    'breach_notification_hours': reg.breach_notification_hours,
                    'penalties': reg.penalties
                }
                for reg in region.regulations
            ],
            'data_residency_required': region.data_residency_required,
            'allowed_data_transfers': region.allowed_data_transfers,
            'quantum_restrictions': region.quantum_computing_restrictions,
            'encryption_standards': region.encryption_standards,
            'audit_requirements': region.audit_requirements
        }
    
    def check_data_processing_compliance(
        self, 
        data_classification: DataClassification,
        processing_location: str,
        quantum_backend: str,
        region_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check compliance for data processing."""
        target_region_id = region_id or self.current_region
        if not target_region_id or target_region_id not in self.regions:
            return {
                'compliant': False,
                'reason': 'No compliance region specified or region not found',
                'requirements': []
            }
        
        region = self.regions[target_region_id]
        applicable_regulations = region.get_applicable_regulations(data_classification)
        
        compliance_issues = []
        requirements = []
        
        # Check data residency requirements
        if region.data_residency_required:
            if processing_location not in region.country_codes:
                compliance_issues.append(f"Data processing location {processing_location} not allowed")
            requirements.append("Data must be processed within region boundaries")
        
        # Check quantum computing restrictions
        for restriction in region.quantum_computing_restrictions:
            if "no_phi_on_public_quantum" in restriction and quantum_backend != "private":
                compliance_issues.append("PHI data cannot be processed on public quantum systems")
            if "government_approval_required" in restriction:
                requirements.append("Government approval required for quantum computing")
        
        # Check regulation-specific requirements
        for reg in applicable_regulations:
            if reg.encryption_required:
                requirements.append(f"Encryption required per {reg.name}")
            
            if reg.audit_logging_required:
                requirements.append(f"Audit logging required per {reg.name}")
            
            if reg.consent_required:
                requirements.append(f"User consent required per {reg.name}")
            
            if reg.retention_period_days:
                requirements.append(f"Data retention limit: {reg.retention_period_days} days per {reg.name}")
        
        return {
            'compliant': len(compliance_issues) == 0,
            'issues': compliance_issues,
            'requirements': requirements,
            'applicable_regulations': [reg.regulation_id for reg in applicable_regulations],
            'region': region.name
        }
    
    def validate_data_transfer(
        self,
        source_region: str,
        destination_region: str,
        data_classification: DataClassification,
        transfer_mechanism: str
    ) -> Dict[str, Any]:
        """Validate data transfer between regions."""
        
        if source_region not in self.regions or destination_region not in self.regions:
            return {
                'allowed': False,
                'reason': 'One or both regions not found',
                'requirements': []
            }
        
        source = self.regions[source_region]
        destination = self.regions[destination_region]
        
        # Same region transfers are generally allowed
        if source_region == destination_region:
            return {
                'allowed': True,
                'reason': 'Same region transfer',
                'requirements': []
            }
        
        # Check if transfer is allowed
        if transfer_mechanism not in source.allowed_data_transfers:
            return {
                'allowed': False,
                'reason': f'Transfer mechanism {transfer_mechanism} not allowed from {source.name}',
                'requirements': source.allowed_data_transfers
            }
        
        # Check destination region requirements
        dest_regulations = destination.get_applicable_regulations(data_classification)
        additional_requirements = []
        
        for reg in dest_regulations:
            if reg.consent_required:
                additional_requirements.append(f"Consent required in destination ({reg.name})")
            if reg.encryption_required:
                additional_requirements.append(f"Encryption required in destination ({reg.name})")
        
        return {
            'allowed': True,
            'reason': 'Transfer allowed with requirements',
            'requirements': additional_requirements,
            'source_region': source.name,
            'destination_region': destination.name
        }
    
    def get_encryption_requirements(self, region_id: Optional[str] = None) -> List[str]:
        """Get encryption requirements for region."""
        target_region_id = region_id or self.current_region
        if target_region_id and target_region_id in self.regions:
            return self.regions[target_region_id].encryption_standards
        return []
    
    def get_audit_requirements(self, region_id: Optional[str] = None) -> Dict[str, Any]:
        """Get audit requirements for region."""
        target_region_id = region_id or self.current_region
        if target_region_id and target_region_id in self.regions:
            return self.regions[target_region_id].audit_requirements
        return {}
    
    def calculate_data_retention_period(
        self,
        data_classification: DataClassification,
        region_id: Optional[str] = None
    ) -> Optional[int]:
        """Calculate required data retention period in days."""
        target_region_id = region_id or self.current_region
        if target_region_id and target_region_id in self.regions:
            region = self.regions[target_region_id]
            return region.get_max_retention_days(data_classification)
        return None
    
    def check_breach_notification_requirements(
        self,
        incident_severity: str,
        data_types_affected: List[str],
        region_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check breach notification requirements."""
        target_region_id = region_id or self.current_region
        if not target_region_id or target_region_id not in self.regions:
            return {'notification_required': False, 'timeframe_hours': None}
        
        region = self.regions[target_region_id]
        notification_requirements = []
        
        for reg in region.regulations:
            # Check if regulation applies to affected data types
            if any(data_type in reg.data_types or 'all' in reg.data_types 
                  for data_type in data_types_affected):
                
                if reg.breach_notification_hours:
                    notification_requirements.append({
                        'regulation': reg.name,
                        'timeframe_hours': reg.breach_notification_hours,
                        'penalties': reg.penalties
                    })
        
        if not notification_requirements:
            return {'notification_required': False, 'timeframe_hours': None}
        
        # Use most restrictive timeframe
        min_timeframe = min(req['timeframe_hours'] for req in notification_requirements)
        
        return {
            'notification_required': True,
            'timeframe_hours': min_timeframe,
            'requirements': notification_requirements
        }
    
    def generate_compliance_report(self, region_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        target_region_id = region_id or self.current_region
        if not target_region_id or target_region_id not in self.regions:
            return {'error': 'No compliance region specified'}
        
        region = self.regions[target_region_id]
        
        report = {
            'region_info': self.get_region_info(target_region_id),
            'compliance_summary': {
                'total_regulations': len(region.regulations),
                'critical_regulations': len([r for r in region.regulations if r.compliance_level == ComplianceLevel.CRITICAL]),
                'data_residency_required': region.data_residency_required,
                'quantum_restrictions': len(region.quantum_computing_restrictions) > 0
            },
            'data_classification_requirements': {},
            'generated_at': datetime.now().isoformat()
        }
        
        # Add requirements by data classification
        for data_class in DataClassification:
            applicable_regs = region.get_applicable_regulations(data_class)
            retention_days = region.get_max_retention_days(data_class)
            
            report['data_classification_requirements'][data_class.value] = {
                'applicable_regulations': [reg.regulation_id for reg in applicable_regs],
                'encryption_required': any(reg.encryption_required for reg in applicable_regs),
                'consent_required': any(reg.consent_required for reg in applicable_regs),
                'audit_logging_required': any(reg.audit_logging_required for reg in applicable_regs),
                'right_to_erasure': any(reg.right_to_erasure for reg in applicable_regs),
                'data_portability': any(reg.data_portability for reg in applicable_regs),
                'max_retention_days': retention_days
            }
        
        return report
    
    def suggest_optimal_region(
        self,
        data_classification: DataClassification,
        user_locations: List[str],
        quantum_backend_type: str = "public"
    ) -> Dict[str, Any]:
        """Suggest optimal compliance region based on requirements."""
        
        suggestions = []
        
        for region_id, region in self.regions.items():
            # Calculate compatibility score
            score = 0
            issues = []
            
            # Check user location compatibility
            location_match = any(loc in region.country_codes for loc in user_locations)
            if location_match:
                score += 30
            
            # Check data residency requirements
            if region.data_residency_required and not location_match:
                score -= 20
                issues.append("Data residency required but user not in region")
            
            # Check quantum computing restrictions
            if region.quantum_computing_restrictions:
                if quantum_backend_type == "public" and "no_phi_on_public_quantum" in region.quantum_computing_restrictions:
                    score -= 30
                    issues.append("Public quantum computing restricted")
                if "government_approval_required" in region.quantum_computing_restrictions:
                    score -= 10
                    issues.append("Government approval required")
            
            # Check regulation complexity
            applicable_regs = region.get_applicable_regulations(data_classification)
            critical_regs = [reg for reg in applicable_regs if reg.compliance_level == ComplianceLevel.CRITICAL]
            
            score -= len(critical_regs) * 5  # Penalty for complexity
            
            suggestions.append({
                'region_id': region_id,
                'region_name': region.name,
                'compatibility_score': score,
                'issues': issues,
                'applicable_regulations': [reg.regulation_id for reg in applicable_regs],
                'critical_regulations': len(critical_regs),
                'data_residency_required': region.data_residency_required
            })
        
        # Sort by compatibility score
        suggestions.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        return {
            'recommended_region': suggestions[0] if suggestions else None,
            'all_options': suggestions,
            'data_classification': data_classification.value,
            'user_locations': user_locations,
            'quantum_backend_type': quantum_backend_type
        }
    
    def export_compliance_config(self, file_path: str):
        """Export compliance configuration to JSON file."""
        try:
            config = {
                'regions': {},
                'current_region': self.current_region,
                'export_timestamp': datetime.now().isoformat()
            }
            
            for region_id, region in self.regions.items():
                config['regions'][region_id] = {
                    'region_id': region.region_id,
                    'name': region.name,
                    'country_codes': region.country_codes,
                    'data_residency_required': region.data_residency_required,
                    'allowed_data_transfers': region.allowed_data_transfers,
                    'quantum_computing_restrictions': region.quantum_computing_restrictions,
                    'encryption_standards': region.encryption_standards,
                    'audit_requirements': region.audit_requirements,
                    'regulations': [
                        {
                            'regulation_id': reg.regulation_id,
                            'name': reg.name,
                            'description': reg.description,
                            'compliance_level': reg.compliance_level.value,
                            'data_types': reg.data_types,
                            'geographic_scope': reg.geographic_scope,
                            'retention_period_days': reg.retention_period_days,
                            'encryption_required': reg.encryption_required,
                            'audit_logging_required': reg.audit_logging_required,
                            'consent_required': reg.consent_required,
                            'right_to_erasure': reg.right_to_erasure,
                            'data_portability': reg.data_portability,
                            'breach_notification_hours': reg.breach_notification_hours,
                            'penalties': reg.penalties
                        }
                        for reg in region.regulations
                    ]
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Compliance configuration exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export compliance configuration: {e}")
            raise
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get overall compliance summary."""
        summary = {
            'current_region': self.current_region,
            'available_regions': list(self.regions.keys()),
            'total_regions': len(self.regions),
            'regions_by_type': {},
            'regulation_summary': {}
        }
        
        # Categorize regions
        for region in self.regions.values():
            for reg in region.regulations:
                if reg.compliance_level.value not in summary['regions_by_type']:
                    summary['regions_by_type'][reg.compliance_level.value] = []
                summary['regions_by_type'][reg.compliance_level.value].append(region.region_id)
        
        # Regulation statistics
        all_regulations = []
        for region in self.regions.values():
            all_regulations.extend(region.regulations)
        
        summary['regulation_summary'] = {
            'total_regulations': len(all_regulations),
            'by_level': {level.value: len([r for r in all_regulations if r.compliance_level == level]) 
                        for level in ComplianceLevel},
            'encryption_required': len([r for r in all_regulations if r.encryption_required]),
            'consent_required': len([r for r in all_regulations if r.consent_required]),
            'right_to_erasure': len([r for r in all_regulations if r.right_to_erasure]),
            'data_portability': len([r for r in all_regulations if r.data_portability])
        }
        
        return summary
