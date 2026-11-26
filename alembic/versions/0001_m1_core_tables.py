"""Create the M1 group message and decision audit tables."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_m1_core_tables"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "group_messages",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("bot_id", sa.Text(), nullable=False),
        sa.Column("group_id", sa.BigInteger(), nullable=False),
        sa.Column("msg_id", sa.Text(), nullable=False),
        sa.Column("sender_qq", sa.BigInteger(), nullable=False),
        sa.Column("sender_alias", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("reply_to_msg_id", sa.Text()),
        sa.Column("is_bot", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "bot_id", "group_id", "msg_id", name="uq_group_messages_bot_group_msgid"
        ),
    )
    op.create_index(
        "idx_group_messages_bot_group_ts",
        "group_messages",
        ["bot_id", "group_id", sa.text("created_at DESC")],
    )

    op.create_table(
        "decision_audit_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("bot_id", sa.Text(), nullable=False),
        sa.Column("group_id", sa.BigInteger(), nullable=False),
        sa.Column("context_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("gate_output", postgresql.JSONB(), nullable=False),
        sa.Column("main_vetoed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("final_reply", sa.Text()),
        sa.Column("stale_action", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("human_label", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "idx_audit_logs_bot_group_ts",
        "decision_audit_logs",
        ["bot_id", "group_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("idx_audit_logs_bot_group_ts", table_name="decision_audit_logs")
    op.drop_table("decision_audit_logs")
    op.drop_index("idx_group_messages_bot_group_ts", table_name="group_messages")
    op.drop_table("group_messages")
