"""Generate synthetic KB articles and eval set for the RAG demo.

Uses a deterministic, template-based approach (seed=42) to produce 200 KB
articles with the data properties required by the demo scenarios:

1. Cross-region error code overlap (8+ codes in 2+ regions)
2. Deprecated documents (~15%, 30 articles) with newer replacements
3. Multi-chunk topics (10+ topics covered by 2-3 docs each)
4. Version-date alignment (v1.0->2023, v2.0->2024, v3.0->2025)
5. Even category distribution (~50 per category)
6. Fixed error code pool (~40 codes, 0-3 per document)

Run:
    uv run python -m data.generate_dataset
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
NUM_ARTICLES = 200
DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "kb_articles.json"
EVAL_OUTPUT_PATH = DATA_DIR.parent / "evals" / "eval_set.json"

REGIONS: list[str] = ["EU", "US", "APAC"]
VERSIONS: list[str] = ["v1.0", "v2.0", "v3.0"]
CATEGORIES: list[str] = ["authentication", "billing", "deployment", "networking"]

# Error code pools per category (15 each = 60 total, spec says ~40 but we
# need room for the cross-region codes like E-4012 referenced in demos)
ERROR_CODES: dict[str, list[str]] = {
    "authentication": [f"E-{1001 + i}" for i in range(15)],
    "billing": [f"E-{2001 + i}" for i in range(15)],
    "deployment": [f"E-{3001 + i}" for i in range(15)],
    "networking": [f"E-{4001 + i}" for i in range(15)],
}

# Version -> date range alignment
VERSION_DATE_RANGES: dict[str, tuple[date, date]] = {
    "v1.0": (date(2023, 1, 1), date(2023, 12, 31)),
    "v2.0": (date(2024, 1, 1), date(2024, 12, 31)),
    "v3.0": (date(2025, 1, 1), date(2025, 6, 1)),
}

# ---------------------------------------------------------------------------
# Cross-region error codes (Property 1)
# At least 8 error codes appear in 2+ regions with *different* resolution steps.
# ---------------------------------------------------------------------------
CROSS_REGION_CODES: list[str] = [
    "E-1001", "E-1005", "E-2003", "E-2007",
    "E-3002", "E-3006", "E-4001", "E-4005",
    "E-4008", "E-4012",
]

REGION_SPECIFIC_RESOLUTION: dict[str, dict[str, str]] = {
    "EU": {
        "E-1001": "Reset credentials through the EU SSO portal at sso.eu.example.com",
        "E-1005": "Contact the EU identity team at eu-identity@example.com for MFA reset",
        "E-2003": "Submit a billing dispute via the EU finance portal",
        "E-2007": "EU invoices require VAT recalculation — use the EU billing dashboard",
        "E-3002": "Use the EU-West-1 deployment pipeline for container re-provisioning",
        "E-3006": "Route through the EU CDN edge node at eu-edge.example.com",
        "E-4001": "Check the EU regional firewall rules at fw.eu.example.com",
        "E-4005": "Reset the EU load balancer via the EU ops console",
        "E-4008": "EU DNS resolution requires updating the eu-dns.example.com zone file",
        "E-4012": "Reset via the EU SSO portal and clear regional session cache",
    },
    "US": {
        "E-1001": "Reset credentials using the admin CLI: `auth-reset --region us`",
        "E-1005": "File a ticket with US-IAM support for MFA bypass",
        "E-2003": "Open a case with US accounts receivable at us-billing@example.com",
        "E-2007": "US invoices use the standard billing portal — no tax recalculation needed",
        "E-3002": "Redeploy using the US-East-1 CI/CD pipeline",
        "E-3006": "Use the US CDN origin at us-origin.example.com",
        "E-4001": "Review the US VPC security groups in the AWS console",
        "E-4005": "Restart the US application load balancer from the US ops dashboard",
        "E-4008": "US DNS issues require updating Route53 hosted zone entries",
        "E-4012": "Reset via the admin CLI and flush the US auth token cache",
    },
    "APAC": {
        "E-1001": "Contact APAC regional support at apac-support@example.com for credential reset",
        "E-1005": "APAC MFA resets go through the APAC security desk",
        "E-2003": "APAC billing disputes are handled by the Singapore finance office",
        "E-2007": "APAC invoices require GST adjustment via the APAC billing module",
        "E-3002": "Use the APAC-Southeast-1 deployment orchestrator",
        "E-3006": "Route through the APAC CDN PoP at apac-cdn.example.com",
        "E-4001": "Check the APAC regional firewall at fw.apac.example.com",
        "E-4005": "Reset the APAC load balancer using the APAC infra portal",
        "E-4008": "APAC DNS managed by the Tokyo zone — update apac-dns.example.com",
        "E-4012": "Contact APAC regional support for session cache and token reset",
    },
}

# ---------------------------------------------------------------------------
# Multi-chunk topic groups (Property 3)
# 10+ topics, each covered by 2-3 documents with partial information.
# ---------------------------------------------------------------------------
MULTI_CHUNK_TOPICS: list[dict[str, Any]] = [
    {
        "topic_group": "sso_setup",
        "category": "authentication",
        "docs": [
            {"title_suffix": "SAML Configuration", "focus": "SAML protocol setup, certificate exchange, and IdP metadata import"},
            {"title_suffix": "IdP Enrollment", "focus": "enrolling users in the identity provider, group mapping, and role assignment"},
            {"title_suffix": "SSO Troubleshooting", "focus": "common SSO login failures, token expiry issues, and debug logging"},
        ],
    },
    {
        "topic_group": "mfa_rollout",
        "category": "authentication",
        "docs": [
            {"title_suffix": "MFA Policy Configuration", "focus": "enabling MFA requirements, policy scopes, and grace periods"},
            {"title_suffix": "MFA Device Enrollment", "focus": "TOTP app setup, hardware key registration, and backup codes"},
        ],
    },
    {
        "topic_group": "invoice_management",
        "category": "billing",
        "docs": [
            {"title_suffix": "Invoice Generation", "focus": "automated invoice creation, billing cycles, and PDF export"},
            {"title_suffix": "Invoice Disputes and Credits", "focus": "disputing charges, requesting credits, and refund workflows"},
            {"title_suffix": "Tax and Compliance", "focus": "VAT/GST handling, tax-exempt status, and compliance documentation"},
        ],
    },
    {
        "topic_group": "subscription_lifecycle",
        "category": "billing",
        "docs": [
            {"title_suffix": "Plan Upgrades and Downgrades", "focus": "changing subscription tiers, prorated billing, and feature access"},
            {"title_suffix": "Subscription Renewal and Cancellation", "focus": "auto-renewal settings, cancellation flow, and data retention"},
        ],
    },
    {
        "topic_group": "container_deployment",
        "category": "deployment",
        "docs": [
            {"title_suffix": "Container Image Build", "focus": "Dockerfile best practices, multi-stage builds, and image scanning"},
            {"title_suffix": "Container Orchestration", "focus": "Kubernetes pod configuration, service mesh setup, and scaling policies"},
            {"title_suffix": "Container Monitoring", "focus": "log aggregation, health checks, and resource utilization dashboards"},
        ],
    },
    {
        "topic_group": "blue_green_deploy",
        "category": "deployment",
        "docs": [
            {"title_suffix": "Blue-Green Setup", "focus": "environment provisioning, traffic routing rules, and DNS switching"},
            {"title_suffix": "Blue-Green Rollback Procedures", "focus": "rollback triggers, state verification, and post-rollback validation"},
        ],
    },
    {
        "topic_group": "vpn_configuration",
        "category": "networking",
        "docs": [
            {"title_suffix": "VPN Gateway Setup", "focus": "IPSec tunnel configuration, gateway provisioning, and key exchange"},
            {"title_suffix": "VPN Client Configuration", "focus": "client software setup, split tunneling, and certificate installation"},
            {"title_suffix": "VPN Troubleshooting", "focus": "connection drops, MTU issues, and route table debugging"},
        ],
    },
    {
        "topic_group": "dns_management",
        "category": "networking",
        "docs": [
            {"title_suffix": "DNS Zone Management", "focus": "creating hosted zones, record types, and TTL best practices"},
            {"title_suffix": "DNS Failover and Health Checks", "focus": "health-check policies, failover routing, and latency-based routing"},
        ],
    },
    {
        "topic_group": "api_gateway",
        "category": "networking",
        "docs": [
            {"title_suffix": "API Gateway Configuration", "focus": "route definitions, rate limiting, and request transformation"},
            {"title_suffix": "API Gateway Authentication", "focus": "API key management, OAuth integration, and JWT validation"},
        ],
    },
    {
        "topic_group": "cert_management",
        "category": "authentication",
        "docs": [
            {"title_suffix": "Certificate Provisioning", "focus": "requesting SSL/TLS certificates, domain validation, and auto-renewal"},
            {"title_suffix": "Certificate Rotation", "focus": "scheduled rotation, zero-downtime swaps, and revocation procedures"},
        ],
    },
]

# ---------------------------------------------------------------------------
# Body paragraph templates per category
# ---------------------------------------------------------------------------
BODY_TEMPLATES: dict[str, list[str]] = {
    "authentication": [
        (
            "This article addresses authentication issues encountered in the {region} region "
            "when running product version {version}. Users may experience login failures, "
            "token expiration errors, or session management problems. The following steps "
            "outline the recommended resolution path for your region and version."
        ),
        (
            "Authentication in {version} uses a token-based flow with regional session "
            "management. When the auth service in {region} encounters a failure, it returns "
            "a structured error response. Check the authentication logs at "
            "/var/log/auth/{region_lower}/ for detailed trace information."
        ),
        (
            "To resolve this issue, first verify that your identity provider configuration "
            "is correct for the {region} region. Ensure that the OAuth callback URL matches "
            "your {version} deployment endpoint. {resolution}"
        ),
        (
            "If the problem persists after following the above steps, escalate to the {region} "
            "regional support team. Include the error code, your tenant ID, and the timestamp "
            "of the failure. The {version} support matrix is available in the internal wiki."
        ),
    ],
    "billing": [
        (
            "This document covers billing operations for the {region} region under product "
            "version {version}. Billing workflows differ by region due to tax regulations, "
            "currency handling, and invoicing requirements."
        ),
        (
            "The billing engine in {version} processes charges on a monthly cycle. For {region} "
            "customers, invoices are generated on the first business day of each month. "
            "Any adjustments or credits must be submitted before the billing window closes."
        ),
        (
            "To address this billing issue, navigate to the {region} billing dashboard and "
            "review the transaction history for the affected period. {resolution} Ensure "
            "that your payment method is current and that no holds are applied."
        ),
        (
            "For unresolved billing discrepancies, contact the {region} finance team. Provide "
            "the invoice number, charge date, and expected amount. The {version} billing API "
            "can also be queried programmatically for audit purposes."
        ),
    ],
    "deployment": [
        (
            "This guide covers deployment procedures for the {region} region using product "
            "version {version}. Deployments in {region} follow a region-specific pipeline "
            "with compliance checks and approval gates."
        ),
        (
            "The {version} deployment pipeline uses infrastructure-as-code templates that "
            "are parameterized per region. For {region}, ensure that the deployment manifest "
            "references the correct regional endpoints and storage buckets."
        ),
        (
            "When encountering deployment failures, first check the CI/CD pipeline logs in "
            "the {region} build system. {resolution} Verify that all pre-deployment checks "
            "have passed, including security scans and integration tests."
        ),
        (
            "For rollback scenarios, the {version} platform supports automated rollback within "
            "a 30-minute window. After that, manual intervention by the {region} ops team "
            "is required. Always maintain a known-good deployment artifact."
        ),
    ],
    "networking": [
        (
            "This article documents networking configuration and troubleshooting for the "
            "{region} region on product version {version}. Network topology varies by region "
            "due to data residency requirements and peering arrangements."
        ),
        (
            "The {version} networking stack uses a layered architecture with regional "
            "gateways, load balancers, and edge nodes. In {region}, traffic is routed through "
            "the regional PoP before reaching the application tier."
        ),
        (
            "To diagnose networking issues, start with connectivity checks from the {region} "
            "monitoring dashboard. {resolution} Use the network diagnostic CLI tool "
            "`netdiag --region {region_lower}` to perform traceroutes and latency tests."
        ),
        (
            "If the issue involves cross-region traffic, verify that the peering links between "
            "{region} and other regions are healthy. The {version} network map is available "
            "in the infrastructure console under the topology viewer."
        ),
    ],
}

# Titles for general (non-cross-region, non-multi-chunk) articles
TITLE_TEMPLATES: dict[str, list[str]] = {
    "authentication": [
        "Resolving {error_code}: Authentication Timeout",
        "Configuring Single Sign-On for {region}",
        "Token Refresh Failures in {version}",
        "Session Management Best Practices",
        "Password Policy Configuration Guide",
        "LDAP Integration Setup for {region}",
        "OAuth 2.0 Flow Troubleshooting",
        "Service Account Authentication Guide",
        "Kerberos Delegation Configuration",
        "Authentication Audit Log Analysis",
        "Multi-Tenant Auth Isolation in {version}",
        "Emergency Credential Rotation Procedure",
    ],
    "billing": [
        "Resolving {error_code}: Billing Discrepancy",
        "Setting Up Auto-Pay for {region}",
        "Usage Metering Configuration in {version}",
        "Credit Application and Refund Process",
        "Enterprise Billing Portal Guide",
        "Cost Allocation Tag Management",
        "Billing API Reference for {version}",
        "Payment Gateway Integration for {region}",
        "Billing Notification Setup",
        "Reserved Capacity Billing Guide",
        "Multi-Currency Support in {region}",
        "Billing Export and Reporting Tools",
    ],
    "deployment": [
        "Resolving {error_code}: Deployment Failure",
        "Rolling Update Strategy for {region}",
        "Canary Deployment Configuration in {version}",
        "Infrastructure Provisioning Guide",
        "Secrets Management During Deployment",
        "Regional Compliance Checks for {region}",
        "Deployment Artifact Management",
        "Environment Variable Configuration",
        "Health Check Setup for {version}",
        "Deployment Quota Management in {region}",
        "Hotfix Deployment Procedure",
        "Deployment Pipeline Optimization",
    ],
    "networking": [
        "Resolving {error_code}: Network Connectivity",
        "Load Balancer Configuration for {region}",
        "Firewall Rule Management in {version}",
        "CDN Configuration and Cache Invalidation",
        "Network Peering Setup Guide",
        "SSL/TLS Certificate Management for {region}",
        "Network ACL Best Practices",
        "Bandwidth Throttling Configuration",
        "Network Monitoring and Alerting in {version}",
        "Cross-Region Traffic Routing in {region}",
        "Network Segmentation Guide",
        "Proxy Configuration for {region}",
    ],
}

# Focused body paragraphs for multi-chunk docs
MULTI_CHUNK_BODY_TEMPLATE: str = (
    "This article focuses specifically on {focus} as part of the broader {topic_group_label} "
    "workflow for the {region} region running {version}.\n\n"
    "{detail_p1}\n\n"
    "{detail_p2}\n\n"
    "For related information, consult the other articles in this topic area. Together "
    "they provide comprehensive coverage of {topic_group_label} for {region} {version}."
)

MULTI_CHUNK_DETAIL_BANK: dict[str, list[str]] = {
    "authentication": [
        "Begin by verifying the identity provider metadata is correctly imported into the {region} tenant. The metadata XML must include the correct entity ID, assertion consumer service URL, and signing certificate. Mismatched certificates are the most common cause of configuration failures.",
        "User enrollment requires mapping external identity groups to internal roles. Use the admin console to create group-to-role mappings. For {region}, ensure that regional compliance roles (data steward, privacy officer) are included in the mapping table.",
        "When troubleshooting, enable verbose logging on the auth service by setting LOG_LEVEL=DEBUG in the {region} configuration. Check for SAML assertion timestamp skew, certificate chain validation failures, and audience restriction mismatches.",
        "MFA policies can be scoped to specific user groups, IP ranges, or risk levels. The {version} policy engine evaluates conditions in priority order. Ensure that emergency access accounts are excluded from MFA enforcement.",
        "Hardware security keys use the FIDO2/WebAuthn protocol. Registration requires a secure context (HTTPS). The {region} deployment must have WebAuthn relay configured at the edge proxy level.",
        "Certificate rotation should be scheduled during maintenance windows. The {version} platform supports dual-certificate mode where both old and new certificates are valid during the transition period.",
    ],
    "billing": [
        "Invoice generation is triggered automatically at the close of each billing period. The system aggregates usage records, applies pricing tiers, and generates a PDF invoice. For {region}, local tax rules are applied based on the customer's billing address.",
        "Dispute resolution follows a structured workflow: submit dispute, provide evidence, review by finance team, and resolution notification. For {region}, disputes must be filed within 90 days of the invoice date.",
        "Tax and compliance documentation must be maintained for audit purposes. The {version} billing module supports automatic tax calculation using regional tax tables. VAT/GST rates are updated quarterly.",
        "Plan changes take effect at the start of the next billing cycle unless immediate activation is requested. Prorated charges are calculated based on the remaining days in the current period.",
        "Cancellation triggers a data retention countdown. Customer data is retained for 30 days after cancellation in {region}, after which it is permanently deleted in compliance with regional regulations.",
    ],
    "deployment": [
        "Docker images should use multi-stage builds to minimize attack surface. The final stage should use a minimal base image. Run security scanning with the built-in image scanner before pushing to the {region} container registry.",
        "Kubernetes deployments in {region} require resource limits, liveness probes, and readiness probes. The {version} platform enforces pod security policies that restrict privileged containers and host network access.",
        "Monitoring dashboards aggregate logs from all pods in the deployment. Use structured logging (JSON format) for consistent parsing. The {region} monitoring stack includes Prometheus metrics and Grafana dashboards.",
        "Blue-green deployments maintain two identical environments. Traffic is switched at the load balancer level. The {version} platform provides a one-click switch with automatic health validation.",
        "Rollback is triggered when health checks fail within the first 5 minutes of a deployment. The {version} orchestrator automatically routes traffic back to the previous environment and sends an alert to the {region} ops channel.",
    ],
    "networking": [
        "IPSec tunnels require matching phase 1 and phase 2 parameters on both endpoints. The {region} gateway uses IKEv2 with AES-256-GCM encryption. Pre-shared keys must be rotated every 90 days.",
        "VPN client configuration varies by operating system. The {version} platform provides pre-configured profiles for Windows, macOS, and Linux. Split tunneling is enabled by default for {region} to optimize bandwidth.",
        "Connection drops are often caused by MTU mismatches. The {region} gateway uses an MTU of 1400. Run `ping -s 1372 -M do gateway.{region_lower}.example.com` to verify path MTU.",
        "DNS zones should be configured with appropriate TTL values. For {region}, use a TTL of 300 seconds for A records and 3600 seconds for NS records. DNSSEC is required for all {version} deployments.",
        "Health checks verify endpoint availability at 30-second intervals. Failed checks trigger automatic failover to the secondary endpoint within 60 seconds. The {region} health check endpoint must respond with HTTP 200.",
        "API gateway routes are defined in YAML configuration. Rate limiting is applied per API key with configurable burst limits. The {version} gateway supports request/response transformation via middleware plugins.",
    ],
}

# ---------------------------------------------------------------------------
# Distractor documents for Demo 1 (dense search confusion)
# Semantically close to the E-4012 EU query concept (authentication timeouts,
# session issues) but with different error codes, so dense search retrieves
# them instead of the exact match while BM25 still finds E-4012.
# ---------------------------------------------------------------------------
DISTRACTOR_DOCS: list[dict[str, Any]] = [
    {
        "title": "Authentication Stack Timeout Troubleshooting",
        "category": "networking",
        "region": "EU",
        "version": "v2.0",
        "error_codes": ["E-4011"],
        "body": (
            "When the authentication stack times out in the EU region, users typically "
            "see intermittent login failures and session drops. The root cause is often "
            "related to the network layer between the authentication proxy and the identity "
            "provider. Check the connection pool settings and ensure that the timeout "
            "threshold is set to at least 30 seconds for EU deployments.\n\n"
            "To diagnose, enable verbose logging on the EU authentication gateway and look "
            "for connection timeout entries in /var/log/net/eu/auth-proxy.log. If timeouts "
            "correlate with peak traffic, consider scaling the authentication stack horizontally "
            "using the EU auto-scaling group configuration."
        ),
    },
    {
        "title": "Session Token Expiry During Authentication",
        "category": "authentication",
        "region": "EU",
        "version": "v3.0",
        "error_codes": ["E-1003"],
        "body": (
            "Session tokens may expire prematurely during the authentication handshake in the "
            "EU region, causing users to see timeout errors. This typically occurs when the "
            "token TTL is shorter than the authentication round-trip time, especially during "
            "cross-region identity federation.\n\n"
            "To resolve, increase the session token TTL in the EU authentication configuration "
            "from the default 60 seconds to 120 seconds. Also verify that the system clock on "
            "the EU auth servers is synchronized via NTP, as clock skew is a common cause of "
            "premature token expiry and authentication timeouts."
        ),
    },
    {
        "title": "Resolving Network Authentication Gateway Timeouts",
        "category": "networking",
        "region": "EU",
        "version": "v2.0",
        "error_codes": ["E-4013"],
        "body": (
            "Network authentication gateway timeouts in the EU region indicate that the "
            "gateway cannot complete the authentication handshake within the configured "
            "timeout window. This is distinct from application-level auth failures — the "
            "network layer itself is failing to route the authentication request.\n\n"
            "Check the EU gateway health dashboard for connection queue depth and latency "
            "metrics. If the gateway is overloaded, redistribute traffic across the EU "
            "edge nodes. Ensure that the authentication backend is reachable from the "
            "gateway subnet and that no firewall rules are blocking the auth traffic."
        ),
    },
    {
        "title": "EU Authentication Service Latency and Timeout Issues",
        "category": "networking",
        "region": "EU",
        "version": "v3.0",
        "error_codes": [],
        "body": (
            "The EU authentication service has known latency characteristics due to the "
            "multi-hop architecture required for GDPR compliance. Authentication requests "
            "traverse the EU privacy proxy before reaching the identity store, adding "
            "15-40ms of latency. Under load, this can push total authentication time "
            "beyond timeout thresholds.\n\n"
            "Recommended mitigations: enable connection keep-alive on the EU auth proxy, "
            "increase the client-side timeout to 45 seconds, and configure retry with "
            "exponential backoff. For persistent timeout issues, check the EU privacy "
            "proxy logs for queue saturation indicators."
        ),
    },
    {
        "title": "Authentication Timeout After Stack Upgrade",
        "category": "networking",
        "region": "US",
        "version": "v3.0",
        "error_codes": ["E-4009"],
        "body": (
            "After upgrading the authentication stack to a new version, users may experience "
            "timeout errors during login. This is commonly caused by stale session caches "
            "that reference the old authentication endpoints. The upgraded stack expects "
            "connections on new ports or paths that the cached configuration does not reflect.\n\n"
            "To fix, clear the authentication session cache on all regional nodes. For US "
            "deployments, run `auth-cache --flush --region us` on the management host. "
            "Verify that the load balancer health checks have been updated to probe the "
            "new authentication endpoints."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_date(rng: random.Random, version: str) -> str:
    """Return a random ISO date within the version's date range."""
    start, end = VERSION_DATE_RANGES[version]
    delta = (end - start).days
    return (start + timedelta(days=rng.randint(0, delta))).isoformat()


def _pick_error_codes(rng: random.Random, category: str, count: int) -> list[str]:
    """Pick *count* random error codes from the category pool."""
    pool = ERROR_CODES[category]
    return sorted(rng.sample(pool, min(count, len(pool))))


def _format_body(
    paragraphs: list[str],
    region: str,
    version: str,
    resolution: str,
) -> str:
    """Format body paragraphs with region/version/resolution substitution."""
    filled = []
    for p in paragraphs:
        filled.append(
            p.format(
                region=region,
                version=version,
                region_lower=region.lower(),
                resolution=resolution,
            )
        )
    return "\n\n".join(filled)


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------
def generate_articles() -> list[dict[str, Any]]:
    """Generate 200 KB articles with all required data properties."""
    rng = random.Random(SEED)
    articles: list[dict[str, Any]] = []
    doc_counter = 1

    def _next_doc_id() -> str:
        nonlocal doc_counter
        did = f"KB-{doc_counter:04d}"
        doc_counter += 1
        return did

    # ------------------------------------------------------------------
    # Phase 1: Cross-region error code articles (Property 1)
    # 10 codes x 3 regions = 30 articles
    # ------------------------------------------------------------------
    for error_code in CROSS_REGION_CODES:
        # Determine category from error code prefix
        code_num = int(error_code.split("-")[1])
        if code_num < 2000:
            category = "authentication"
        elif code_num < 3000:
            category = "billing"
        elif code_num < 4000:
            category = "deployment"
        else:
            category = "networking"

        for region in REGIONS:
            version = rng.choice(VERSIONS)
            resolution = REGION_SPECIFIC_RESOLUTION[region][error_code]
            title = f"Resolving {error_code}: {category.title()} Issue in {region}"

            body_templates = BODY_TEMPLATES[category]
            # Pick 2-3 paragraphs
            num_paragraphs = rng.randint(2, 4)
            selected = rng.sample(body_templates, min(num_paragraphs, len(body_templates)))
            body = _format_body(selected, region, version, resolution)

            articles.append({
                "doc_id": _next_doc_id(),
                "title": title,
                "body": body,
                "region": region,
                "product_version": version,
                "effective_date": _random_date(rng, version),
                "error_codes": [error_code],
                "category": category,
                "deprecated": False,
                "topic_group": None,
            })

    # ------------------------------------------------------------------
    # Phase 1b: Distractor documents for dense search confusion
    # These are semantically close to the Demo 1 query but have different
    # error codes, ensuring dense search ranks them above the true match.
    # ------------------------------------------------------------------
    for distractor in DISTRACTOR_DOCS:
        articles.append({
            "doc_id": _next_doc_id(),
            "title": distractor["title"],
            "body": distractor["body"],
            "region": distractor["region"],
            "product_version": distractor["version"],
            "effective_date": _random_date(rng, distractor["version"]),
            "error_codes": distractor["error_codes"],
            "category": distractor["category"],
            "deprecated": False,
            "topic_group": None,
        })

    # ------------------------------------------------------------------
    # Phase 2: Deprecated articles with replacements (Property 2)
    # 30 deprecated articles -> 30 docs. Their replacements are generated
    # later as regular articles that reference the same topic.
    # Actually: create 15 deprecated + 15 replacement pairs = 30 docs.
    # We need 30 *deprecated* docs total (~15%), so do 30 pairs = 60 docs,
    # but that's a lot. Instead: 30 deprecated docs, and their replacements
    # come from other phases or are created here as 30 non-deprecated docs.
    # ------------------------------------------------------------------
    deprecated_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i in range(30):
        category = CATEGORIES[i % 4]
        region = REGIONS[i % 3]
        # Deprecated doc is v1.0 or v2.0, replacement is one version newer
        old_version = "v1.0" if i % 2 == 0 else "v2.0"
        new_version = "v2.0" if old_version == "v1.0" else "v3.0"

        error_code_count = rng.randint(0, 2)
        codes = _pick_error_codes(rng, category, error_code_count)

        base_title_pool = TITLE_TEMPLATES[category]
        title_template = base_title_pool[i % len(base_title_pool)]
        code_for_title = codes[0] if codes else f"E-{1001 + i % 10}"
        old_title = title_template.format(
            error_code=code_for_title, region=region, version=old_version
        )
        new_title = old_title.replace(old_version, new_version)
        if old_title == new_title:
            new_title = f"[Updated] {old_title}"

        body_templates = BODY_TEMPLATES[category]
        selected = rng.sample(body_templates, rng.randint(2, 3))
        old_body = _format_body(
            selected, region, old_version,
            "Follow the standard resolution procedure for this version."
        )
        new_body = _format_body(
            selected, region, new_version,
            "This updated procedure replaces the previous version's workflow."
        )

        deprecated_doc = {
            "doc_id": _next_doc_id(),
            "title": old_title,
            "body": old_body,
            "region": region,
            "product_version": old_version,
            "effective_date": _random_date(rng, old_version),
            "error_codes": codes,
            "category": category,
            "deprecated": True,
            "topic_group": None,
        }
        replacement_doc = {
            "doc_id": _next_doc_id(),
            "title": new_title,
            "body": new_body,
            "region": region,
            "product_version": new_version,
            "effective_date": _random_date(rng, new_version),
            "error_codes": codes,
            "category": category,
            "deprecated": False,
            "topic_group": None,
        }
        articles.append(deprecated_doc)
        articles.append(replacement_doc)
        deprecated_pairs.append((deprecated_doc, replacement_doc))

    # ------------------------------------------------------------------
    # Phase 3: Multi-chunk topic articles (Property 3)
    # 10 topics x 2-3 docs each
    # ------------------------------------------------------------------
    for topic in MULTI_CHUNK_TOPICS:
        topic_group = topic["topic_group"]
        category = topic["category"]
        topic_group_label = topic_group.replace("_", " ")

        for doc_spec in topic["docs"]:
            region = rng.choice(REGIONS)
            version = rng.choice(VERSIONS)
            title = f"{doc_spec['title_suffix']} — {topic_group_label.title()} Guide"

            # Build body from focused template
            detail_pool = MULTI_CHUNK_DETAIL_BANK[category]
            detail_paragraphs = rng.sample(detail_pool, min(2, len(detail_pool)))
            body = MULTI_CHUNK_BODY_TEMPLATE.format(
                focus=doc_spec["focus"],
                topic_group_label=topic_group_label,
                region=region,
                version=version,
                detail_p1=detail_paragraphs[0].format(
                    region=region, version=version, region_lower=region.lower()
                ),
                detail_p2=detail_paragraphs[1].format(
                    region=region, version=version, region_lower=region.lower()
                ),
            )

            error_code_count = rng.randint(0, 1)
            codes = _pick_error_codes(rng, category, error_code_count)

            articles.append({
                "doc_id": _next_doc_id(),
                "title": title,
                "body": body,
                "region": region,
                "product_version": version,
                "effective_date": _random_date(rng, version),
                "error_codes": codes,
                "category": category,
                "deprecated": False,
                "topic_group": topic_group,
            })

    # ------------------------------------------------------------------
    # Phase 4: Fill remaining articles to reach 200
    # Distribute evenly across categories and regions
    # ------------------------------------------------------------------
    remaining = NUM_ARTICLES - len(articles)
    cat_counts = {c: sum(1 for a in articles if a["category"] == c) for c in CATEGORIES}

    for i in range(remaining):
        # Pick the category with the fewest articles so far
        category = min(CATEGORIES, key=lambda c: cat_counts[c])
        region = REGIONS[i % 3]
        version = VERSIONS[i % 3]

        error_code_count = rng.randint(0, 3)
        codes = _pick_error_codes(rng, category, error_code_count)

        title_pool = TITLE_TEMPLATES[category]
        title_template = title_pool[i % len(title_pool)]
        code_for_title = codes[0] if codes else f"E-{1001 + (i * 7) % 40}"
        title = title_template.format(
            error_code=code_for_title, region=region, version=version
        )

        body_templates = BODY_TEMPLATES[category]
        num_paragraphs = rng.randint(2, 4)
        selected = rng.sample(body_templates, min(num_paragraphs, len(body_templates)))
        resolution = "Follow the documented resolution steps for your environment."
        body = _format_body(selected, region, version, resolution)

        articles.append({
            "doc_id": _next_doc_id(),
            "title": title,
            "body": body,
            "region": region,
            "product_version": version,
            "effective_date": _random_date(rng, version),
            "error_codes": codes,
            "category": category,
            "deprecated": False,
            "topic_group": None,
        })
        cat_counts[category] += 1

    return articles


def generate_eval_set(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate 20 eval queries referencing real doc_ids from the dataset.

    Categories:
      - exact_match (5): specific error code + region
      - scoped (5): needs correct region/version/deprecation filter
      - multi_doc (4): answer spans multiple docs
      - broad (3): vague query, no filters
      - deprecated_trap (3): query that would match a deprecated doc
    """
    eval_queries: list[dict[str, Any]] = []
    query_counter = 1

    def _next_query_id() -> str:
        nonlocal query_counter
        qid = f"Q-{query_counter:03d}"
        query_counter += 1
        return qid

    # Index helpers
    by_error_region: dict[tuple[str, str], list[dict]] = {}
    by_topic_group: dict[str, list[dict]] = {}
    deprecated_docs: list[dict] = []
    non_deprecated: list[dict] = []

    for a in articles:
        if a["deprecated"]:
            deprecated_docs.append(a)
        else:
            non_deprecated.append(a)
        for code in a["error_codes"]:
            key = (code, a["region"])
            by_error_region.setdefault(key, []).append(a)
        if a.get("topic_group"):
            by_topic_group.setdefault(a["topic_group"], []).append(a)

    # --- exact_match (5) ---
    # Pick cross-region codes where we know the exact region doc
    exact_match_picks = [
        ("E-4012", "EU"), ("E-1001", "US"), ("E-2003", "APAC"),
        ("E-3002", "EU"), ("E-4005", "US"),
    ]
    for code, region in exact_match_picks:
        matches = [
            a for a in by_error_region.get((code, region), [])
            if not a["deprecated"]
        ]
        if matches:
            doc = matches[0]
            eval_queries.append({
                "query_id": _next_query_id(),
                "query": f"How do I fix error {code} in the {region} region?",
                "expected_doc_ids": [doc["doc_id"]],
                "expected_filters": {"region": region, "error_codes": code},
                "category": "exact_match",
            })

    # --- scoped (5) ---
    scoped_specs = [
        {"region": "EU", "version": "v3.0", "category": "authentication"},
        {"region": "US", "version": "v2.0", "category": "billing"},
        {"region": "APAC", "version": "v3.0", "category": "deployment"},
        {"region": "EU", "version": "v2.0", "category": "networking"},
        {"region": "US", "version": "v3.0", "category": "authentication"},
    ]
    for spec in scoped_specs:
        matches = [
            a for a in non_deprecated
            if a["region"] == spec["region"]
            and a["product_version"] == spec["version"]
            and a["category"] == spec["category"]
        ]
        if matches:
            doc = matches[0]
            eval_queries.append({
                "query_id": _next_query_id(),
                "query": (
                    f"What are the {spec['category']} procedures for "
                    f"{spec['region']} on {spec['version']}?"
                ),
                "expected_doc_ids": [doc["doc_id"]],
                "expected_filters": {
                    "region": spec["region"],
                    "product_version": spec["version"],
                    "deprecated": False,
                },
                "category": "scoped",
            })

    # --- multi_doc (4) ---
    multi_doc_topics = ["sso_setup", "vpn_configuration", "invoice_management", "container_deployment"]
    for topic in multi_doc_topics:
        docs = by_topic_group.get(topic, [])
        if docs:
            topic_label = topic.replace("_", " ")
            eval_queries.append({
                "query_id": _next_query_id(),
                "query": f"Give me a complete guide on {topic_label}.",
                "expected_doc_ids": [d["doc_id"] for d in docs],
                "expected_filters": {},
                "category": "multi_doc",
            })

    # --- broad (3) ---
    broad_queries = [
        "What are the best practices for deployment?",
        "How does billing work?",
        "Tell me about networking configuration.",
    ]
    broad_categories = ["deployment", "billing", "networking"]
    for query_text, cat in zip(broad_queries, broad_categories):
        matches = [a for a in non_deprecated if a["category"] == cat][:3]
        eval_queries.append({
            "query_id": _next_query_id(),
            "query": query_text,
            "expected_doc_ids": [m["doc_id"] for m in matches],
            "expected_filters": {},
            "category": "broad",
        })

    # --- deprecated_trap (3) ---
    # Find deprecated docs and expect the non-deprecated replacement
    dep_trap_count = 0
    seen_categories: set[str] = set()
    for dep in deprecated_docs:
        if dep_trap_count >= 3:
            break
        if dep["category"] in seen_categories:
            continue
        # Find a non-deprecated doc with similar category + region
        replacements = [
            a for a in non_deprecated
            if a["category"] == dep["category"]
            and a["region"] == dep["region"]
            and not a["deprecated"]
        ]
        if replacements:
            repl = replacements[0]
            eval_queries.append({
                "query_id": _next_query_id(),
                "query": (
                    f"How do I handle {dep['category']} issues in "
                    f"{dep['region']}? I need current documentation."
                ),
                "expected_doc_ids": [repl["doc_id"]],
                "expected_filters": {
                    "region": dep["region"],
                    "deprecated": False,
                },
                "category": "deprecated_trap",
            })
            seen_categories.add(dep["category"])
            dep_trap_count += 1

    return eval_queries


def validate_articles(articles: list[dict[str, Any]]) -> None:
    """Validate all 6 required data properties. Raises ValueError on failure."""
    # Basic count
    assert len(articles) == NUM_ARTICLES, (
        f"Expected {NUM_ARTICLES} articles, got {len(articles)}"
    )

    # Property 1: Cross-region error code overlap
    code_regions: dict[str, set[str]] = {}
    for a in articles:
        for code in a["error_codes"]:
            code_regions.setdefault(code, set()).add(a["region"])
    cross_region_count = sum(1 for regions in code_regions.values() if len(regions) >= 2)
    assert cross_region_count >= 8, (
        f"Cross-region overlap: {cross_region_count} codes in 2+ regions (need >= 8)"
    )

    # Property 2: Deprecated ratio ~15%
    deprecated_count = sum(1 for a in articles if a["deprecated"])
    ratio = deprecated_count / len(articles)
    assert 0.10 <= ratio <= 0.20, (
        f"Deprecated ratio: {ratio:.2%} ({deprecated_count} docs) — expected 10-20%"
    )

    # Property 3: Multi-chunk topics
    topic_groups = {}
    for a in articles:
        tg = a.get("topic_group")
        if tg:
            topic_groups.setdefault(tg, []).append(a["doc_id"])
    multi_chunk_count = sum(1 for docs in topic_groups.values() if len(docs) >= 2)
    assert multi_chunk_count >= 10, (
        f"Multi-chunk topics: {multi_chunk_count} (need >= 10)"
    )

    # Property 4: Version-date alignment
    for a in articles:
        version = a["product_version"]
        eff_date = date.fromisoformat(a["effective_date"])
        start, end = VERSION_DATE_RANGES[version]
        assert start <= eff_date <= end, (
            f"{a['doc_id']}: date {eff_date} outside range for {version} ({start}–{end})"
        )

    # Property 5: Category distribution (50 each, +/- 10)
    cat_counts = {c: 0 for c in CATEGORIES}
    for a in articles:
        cat_counts[a["category"]] += 1
    for cat, count in cat_counts.items():
        assert 40 <= count <= 60, (
            f"Category '{cat}' has {count} articles (expected 40-60)"
        )

    # Property 6: Error code pool
    all_valid_codes = set()
    for codes in ERROR_CODES.values():
        all_valid_codes.update(codes)
    for a in articles:
        for code in a["error_codes"]:
            assert code in all_valid_codes, (
                f"{a['doc_id']} has invalid error code: {code}"
            )
        assert len(a["error_codes"]) <= 3, (
            f"{a['doc_id']} has {len(a['error_codes'])} error codes (max 3)"
        )


def main() -> None:
    """Generate dataset and eval set, validate, and write to disk."""
    print("Generating 200 KB articles...")
    articles = generate_articles()

    print("Validating data properties...")
    validate_articles(articles)
    print("  All 6 data properties validated successfully.")

    # Remove topic_group from final output (bookkeeping only, not indexed)
    # Actually keep it in the JSON for eval set generation reference,
    # but note it is NOT indexed in ChromaDB.
    output_articles = []
    for a in articles:
        doc = dict(a)
        # Keep topic_group in the output for bookkeeping
        output_articles.append(doc)

    print(f"Writing {len(output_articles)} articles to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_articles, f, indent=2)

    print("Generating eval set (20 queries)...")
    eval_set = generate_eval_set(articles)
    print(f"  Generated {len(eval_set)} eval queries.")

    EVAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing eval set to {EVAL_OUTPUT_PATH}")
    with open(EVAL_OUTPUT_PATH, "w") as f:
        json.dump(eval_set, f, indent=2)

    # Summary
    cat_counts = {c: 0 for c in CATEGORIES}
    for a in articles:
        cat_counts[a["category"]] += 1
    dep_count = sum(1 for a in articles if a["deprecated"])
    topic_groups = set(a.get("topic_group") for a in articles if a.get("topic_group"))

    print("\n--- Dataset Summary ---")
    print(f"Total articles: {len(articles)}")
    print(f"Category distribution: {cat_counts}")
    print(f"Deprecated articles: {dep_count} ({dep_count/len(articles):.1%})")
    print(f"Multi-chunk topic groups: {len(topic_groups)}")

    code_regions: dict[str, set[str]] = {}
    for a in articles:
        for code in a["error_codes"]:
            code_regions.setdefault(code, set()).add(a["region"])
    cross_region = sum(1 for r in code_regions.values() if len(r) >= 2)
    print(f"Cross-region error codes: {cross_region}")

    eval_cats = {}
    for q in eval_set:
        eval_cats[q["category"]] = eval_cats.get(q["category"], 0) + 1
    print(f"Eval queries by category: {eval_cats}")
    print("Done.")


if __name__ == "__main__":
    main()
