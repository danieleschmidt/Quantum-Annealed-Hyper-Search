# Terraform Infrastructure for Quantum Hyper Search
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

# VPC and Networking
resource "aws_vpc" "quantum_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "quantum-hyper-search-vpc"
    Environment = "production"
  }
}

resource "aws_subnet" "public_subnet" {
  count             = 2
  vpc_id            = aws_vpc.quantum_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name = "quantum-public-subnet-${count.index + 1}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "quantum_igw" {
  vpc_id = aws_vpc.quantum_vpc.id
  
  tags = {
    Name = "quantum-internet-gateway"
  }
}

# Load Balancer
resource "aws_lb" "quantum_alb" {
  name               = "quantum-hyper-search-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.public_subnet[*].id
  
  enable_deletion_protection = false
  
  tags = {
    Name = "quantum-application-load-balancer"
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "quantum_asg" {
  name                = "quantum-hyper-search-asg"
  vpc_zone_identifier = aws_subnet.public_subnet[*].id
  target_group_arns   = [aws_lb_target_group.quantum_tg.arn]
  health_check_type   = "ELB"
  
  min_size         = 2
  max_size         = 10
  desired_capacity = 2
  
  launch_template {
    id      = aws_launch_template.quantum_lt.id
    version = "$Latest"
  }
  
  tag {
    key                 = "Name"
    value               = "quantum-hyper-search-instance"
    propagate_at_launch = true
  }
}

# Launch Template
resource "aws_launch_template" "quantum_lt" {
  name_prefix   = "quantum-hyper-search-"
  image_id      = data.aws_ami.amazon_linux.id
  instance_type = "c5.large"
  
  vpc_security_group_ids = [aws_security_group.instance_sg.id]
  
  user_data = base64encode(templatefile("user_data.sh", {}))
  
  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "quantum-hyper-search-instance"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Security Groups
resource "aws_security_group" "alb_sg" {
  name        = "quantum-alb-security-group"
  description = "Security group for quantum application load balancer"
  vpc_id      = aws_vpc.quantum_vpc.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "instance_sg" {
  name        = "quantum-instance-security-group"
  description = "Security group for quantum application instances"
  vpc_id      = aws_vpc.quantum_vpc.id
  
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Target Group
resource "aws_lb_target_group" "quantum_tg" {
  name     = "quantum-hyper-search-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.quantum_vpc.id
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }
}

# Output values
output "load_balancer_dns" {
  value = aws_lb.quantum_alb.dns_name
}

output "vpc_id" {
  value = aws_vpc.quantum_vpc.id
}
