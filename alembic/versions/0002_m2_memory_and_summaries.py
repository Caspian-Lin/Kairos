"""Create M2 summaries, impressions, notes, and audit action columns."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002_m2_memory_and_summaries"
down_revision: str = "0001_m1_core_tables"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.create_table(
        "context_summaries",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("bot_id", sa.Text(), nullable=False),
        sa.Column("group_id", sa.BigInteger(), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("covered_until_msg_id", sa.Text(), nullable=False),
        sa.Column("covered_until_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "bot_id", "group_id", "covered_until_msg_id", name="uq_summary_watermark"
        ),
    )
    op.create_index(
        "idx_context_summaries_bot_group_covered",
        "context_summaries",
        ["bot_id", "group_id", sa.text("covered_until_at DESC")],
    )

    op.create_table(
        "user_impressions",
        sa.Column("bot_id", sa.Text(), primary_key=True),
        sa.Column("group_id", sa.BigInteger(), primary_key=True),
        sa.Column("user_qq", sa.BigInteger(), primary_key=True),
        sa.Column("nickname", sa.Text(), nullable=False),
        sa.Column("impression_summary", sa.Text(), nullable=False),
        sa.Column("affinity_score", postgresql.REAL(), nullable=False, server_default="0.0"),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )

    op.create_table(
        "memory_notes",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("bot_id", sa.Text(), nullable=False),
        sa.Column("scope_group_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "entity_qqs", postgresql.ARRAY(sa.BigInteger()), nullable=False, server_default="{}"
        ),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("keywords", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("importance", sa.SmallInteger(), nullable=False, server_default="1"),
        sa.Column("last_recalled_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "idx_memory_notes_keywords", "memory_notes", ["keywords"], postgresql_using="gin"
    )
    op.create_index(
        "idx_memory_notes_summary_trgm",
        "memory_notes",
        ["summary_text"],
        postgresql_using="gin",
        postgresql_ops={"summary_text": "gin_trgm_ops"},
    )
    op.create_index(
        "idx_memory_notes_scope",
        "memory_notes",
        ["bot_id", "scope_group_id", sa.text("importance DESC"), sa.text("created_at DESC")],
    )

    for column in ("action_type", "target_msg_id", "emoji_id"):
        op.add_column("decision_audit_logs", sa.Column(column, sa.Text()))


def downgrade() -> None:
    for column in ("emoji_id", "target_msg_id", "action_type"):
        op.drop_column("decision_audit_logs", column)

    op.drop_index("idx_memory_notes_scope", table_name="memory_notes")
    op.drop_index("idx_memory_notes_summary_trgm", table_name="memory_notes")
    op.drop_index("idx_memory_notes_keywords", table_name="memory_notes")
    op.drop_table("memory_notes")
    op.drop_table("user_impressions")
    op.drop_index("idx_context_summaries_bot_group_covered", table_name="context_summaries")
    op.drop_table("context_summaries")
