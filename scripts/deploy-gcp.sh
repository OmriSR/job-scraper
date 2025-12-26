#!/bin/bash
# MatchAI GCP Deployment Script
# This script deploys the MatchAI scheduled job to Google Cloud Platform.
#
# Prerequisites:
# 1. gcloud CLI installed and authenticated
# 2. Docker installed and running
# 3. Supabase project created with DATABASE_URL
# 4. Pinecone index created with PINECONE_API_KEY
# 5. Groq API key (GROQ_API_KEY)

set -e

# Configuration - Update these values
PROJECT_ID="${GCP_PROJECT_ID:-matchai-482219}"
REGION="${GCP_REGION:-us-central1}"
JOB_NAME="matchai-job"
SCHEDULER_NAME="matchai-scheduler"
SERVICE_ACCOUNT="matchai-invoker"
REPOSITORY="matchai"
IMAGE_NAME="job"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Install from: https://docs.docker.com/get-docker/"
        exit 1
    fi

    log_info "Prerequisites OK"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required GCP APIs..."

    gcloud services enable \
        run.googleapis.com \
        artifactregistry.googleapis.com \
        cloudscheduler.googleapis.com \
        secretmanager.googleapis.com \
        --project="$PROJECT_ID"

    log_info "APIs enabled"
}

# Create Artifact Registry repository
create_repository() {
    log_info "Creating Artifact Registry repository..."

    if gcloud artifacts repositories describe "$REPOSITORY" \
        --location="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        log_warn "Repository already exists, skipping"
    else
        gcloud artifacts repositories create "$REPOSITORY" \
            --repository-format=docker \
            --location="$REGION" \
            --project="$PROJECT_ID"
        log_info "Repository created"
    fi
}

# Create secrets in Secret Manager
create_secrets() {
    log_info "Setting up secrets in Secret Manager..."
    log_warn "You'll need to manually add secret values via Cloud Console or gcloud CLI"

    local secrets=("groq-api-key" "database-url" "pinecone-api-key" "email-sender" "email-recipient" "email-app-password")

    for secret in "${secrets[@]}"; do
        if gcloud secrets describe "$secret" --project="$PROJECT_ID" &> /dev/null; then
            log_warn "Secret '$secret' already exists"
        else
            echo -n "placeholder" | gcloud secrets create "$secret" \
                --data-file=- \
                --project="$PROJECT_ID"
            log_info "Created secret '$secret' (update with actual value)"
        fi
    done

    echo ""
    log_warn "Update secrets with actual values:"
    echo "  gcloud secrets versions add groq-api-key --data-file=- <<< 'your-groq-key'"
    echo "  gcloud secrets versions add database-url --data-file=- <<< 'your-supabase-url'"
    echo "  gcloud secrets versions add pinecone-api-key --data-file=- <<< 'your-pinecone-key'"
    echo "  gcloud secrets versions add email-sender --data-file=- <<< 'your-gmail@gmail.com'"
    echo "  gcloud secrets versions add email-recipient --data-file=- <<< 'recipient@example.com'"
    echo "  gcloud secrets versions add email-app-password --data-file=- <<< 'xxxx-xxxx-xxxx-xxxx'"
}

# Build and push Docker image
build_and_push() {
    log_info "Building and pushing Docker image..."

    local image_uri="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest"

    # Configure Docker for Artifact Registry
    gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

    # Build image for linux/amd64 (required by Cloud Run)
    docker build --platform linux/amd64 -t "$image_uri" .

    # Push image
    docker push "$image_uri"

    log_info "Image pushed: $image_uri"
}

# Create service account
create_service_account() {
    log_info "Setting up service account..."

    local sa_email="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"

    if gcloud iam service-accounts describe "$sa_email" --project="$PROJECT_ID" &> /dev/null; then
        log_warn "Service account already exists"
    else
        gcloud iam service-accounts create "$SERVICE_ACCOUNT" \
            --display-name="MatchAI Job Invoker" \
            --project="$PROJECT_ID"
        log_info "Service account created"
    fi

    # Grant Cloud Run invoker role
    gcloud run jobs add-iam-policy-binding "$JOB_NAME" \
        --member="serviceAccount:$sa_email" \
        --role="roles/run.invoker" \
        --region="$REGION" \
        --project="$PROJECT_ID" 2>/dev/null || true

    # Grant secret accessor role
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$sa_email" \
        --role="roles/secretmanager.secretAccessor" \
        --condition=None 2>/dev/null || true
}

# Grant secret access to default compute service account
grant_secret_access() {
    log_info "Granting secret access to default compute service account..."

    local project_number=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
    local default_sa="${project_number}-compute@developer.gserviceaccount.com"

    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$default_sa" \
        --role="roles/secretmanager.secretAccessor" \
        --condition=None --quiet 2>/dev/null || true

    log_info "Secret access granted"
}

# Create Cloud Run Job
create_job() {
    log_info "Creating Cloud Run Job..."

    local image_uri="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest"

    local secrets="GROQ_API_KEY=groq-api-key:latest,DATABASE_URL=database-url:latest,PINECONE_API_KEY=pinecone-api-key:latest,EMAIL_SENDER=email-sender:latest,EMAIL_RECIPIENT=email-recipient:latest,EMAIL_APP_PASSWORD=email-app-password:latest"

    # Check if job already exists
    if gcloud run jobs describe "$JOB_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        log_info "Job exists, updating..."
        gcloud run jobs update "$JOB_NAME" \
            --image="$image_uri" \
            --region="$REGION" \
            --project="$PROJECT_ID" \
            --memory="2Gi" \
            --cpu="2" \
            --max-retries="1" \
            --task-timeout="30m" \
            --set-secrets="$secrets"
    else
        log_info "Creating new job..."
        gcloud run jobs create "$JOB_NAME" \
            --image="$image_uri" \
            --region="$REGION" \
            --project="$PROJECT_ID" \
            --memory="2Gi" \
            --cpu="2" \
            --max-retries="1" \
            --task-timeout="30m" \
            --set-secrets="$secrets"
    fi

    log_info "Cloud Run Job created/updated"
}

# Create Cloud Scheduler
create_scheduler() {
    log_info "Creating Cloud Scheduler..."

    local sa_email="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
    local job_uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run"

    if gcloud scheduler jobs describe "$SCHEDULER_NAME" \
        --location="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        log_warn "Scheduler already exists, updating..."
        gcloud scheduler jobs update http "$SCHEDULER_NAME" \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --schedule="0 8,20 * * *" \
            --uri="$job_uri" \
            --http-method="POST" \
            --oauth-service-account-email="$sa_email"
    else
        gcloud scheduler jobs create http "$SCHEDULER_NAME" \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --schedule="0 8,20 * * *" \
            --uri="$job_uri" \
            --http-method="POST" \
            --oauth-service-account-email="$sa_email"
    fi

    log_info "Cloud Scheduler configured to run at 8:00 AM and 8:00 PM daily"
}

# Run job manually
run_job() {
    log_info "Running job manually..."

    gcloud run jobs execute "$JOB_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --wait

    log_info "Job execution complete"
}

# Main deployment flow
deploy() {
    echo ""
    echo "=========================================="
    echo "  MatchAI GCP Deployment"
    echo "=========================================="
    echo ""

    check_prerequisites
    enable_apis
    create_repository
    create_secrets
    grant_secret_access
    build_and_push
    create_job
    create_service_account
    create_scheduler

    echo ""
    log_info "Deployment complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Update secrets with actual values (see commands above)"
    echo "  2. Upload your CV locally:  matchai upload-cv --cv resume.pdf"
    echo "  3. Add companies locally:   matchai add-company --name X --uid Y --token Z"
    echo "  4. Test the job manually:   ./scripts/deploy-gcp.sh run"
    echo "  5. Check results:           matchai get-results"
    echo ""
}

# Parse command
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    run)
        run_job
        ;;
    build)
        build_and_push
        ;;
    *)
        echo "Usage: $0 {deploy|run|build}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Full deployment (default)"
        echo "  run     - Execute job manually"
        echo "  build   - Build and push Docker image only"
        exit 1
        ;;
esac
